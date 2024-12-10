import math
import numpy as np
import pickle
import random
import time
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import BaseModelOutput
import torch.distributed.rpc as rpc
from torch.profiler import profiler, record_function, ProfilerActivity

def monitor_gpu():
    if torch.cuda.is_available():
        print("\nGPU Status:")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        
class BertBackOri(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = "cuda:0"

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
        
class BertBackFFN(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = "cuda:0"

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
        
class BertBackDecomposition(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = "cuda:0"

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        weight = split_layer.intermediate.dense.weight
        bias = split_layer.intermediate.dense.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=1, out_features=1, bias=True)

        self.dense_u.weight = nn.Parameter(u[:, :rank].clone())
        self.dense_u.bias = bias

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.dense_u(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)  # Query: This implementation only splits the FFN layer into FFN-1 and FFN-2 (rather than 3 as described in the paper)?
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
        
class BertBackOriQuant(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = "cuda:0"

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        hidden_states = torch.dequantize(hidden_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
        
class BertBackFFNQuant(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = "cuda:0"

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        hidden_states = torch.dequantize(hidden_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
        
class BertBackFFNQuantRes(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = "cuda:0"

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states_tuple, attention_mask=None):
        ffn_output_q, residual_q = hidden_states_tuple
        ffn_output = torch.dequantize(ffn_output_q).to(self.device)
        residual = torch.dequantize(residual_q).to(self.device)
        
        # Format attention mask properly
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        hidden_states = self.intermediate_act_fn(ffn_output)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
        
class CloudWorker:
    def __init__(self):
        self.model = None
        print("CloudWorker initialised")

    def initialize_model(self, pretrain_dir, split_layer, model_type):
        print(f"\nInitialising model type: {model_type}")
        model_classes = {
            1: BertBackOri,
            2: BertBackFFN,
            3: BertBackDecomposition,
            4: BertBackOriQuant,
            5: BertBackFFNQuant,
            6: BertBackFFNQuantRes
        }
        
        device = "cuda:0"
        rank = 8  # For decomposition model
        label_nums = 2  # For SST-2 binary classification
        
        self.model = model_classes[model_type](
            pretrain_dir=pretrain_dir,
            split_num=split_layer,
            rank=rank,
            label_nums=label_nums,
            device=device
        )
        print(f"Model is on device: {next(self.model.parameters()).device}")
        return True

    def forward(self, hidden_states, attn_mask=None):
        if self.model is None:
            raise RuntimeError("Model not initialised")
        
        if isinstance(hidden_states, tuple):
            hidden_states = (hidden_states[0].to("cuda:0"), hidden_states[1].to("cuda:0"))
        else:
            hidden_states = hidden_states.to("cuda:0")
        
        outputs = self.model(hidden_states, attn_mask)
        torch.cuda.synchronize()
        return outputs
        
    def parameter_rrefs(self):
        if self.model is None:
            raise RuntimeError("Model not initialised")
        return [rpc.RRef(p) for p in self.model.parameters()]

    def load_state_dict(self, state_dict):
        if self.model is None:
            raise RuntimeError("Model not initialised")
        self.model.load_state_dict(state_dict)
        self.model.to("cuda:0")
        return True
    
    def receive_tensor(self, hidden_states):
        return hidden_states