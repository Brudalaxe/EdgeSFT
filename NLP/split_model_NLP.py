import os

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from transformers import BertModel, BertTokenizerFast
from model_classes_NLP import BertBackOri, BertBackFFN, BertBackDecomposition, BertBackOriQuant, BertBackFFNQuant, BertBackFFNQuantRes, CloudWorker

class BertFrontOri(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num)])

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class BertFrontFFN(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention
        self.dense = split_layer.intermediate.dense

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.attention(hidden_states, extended_attention_mask)[0]
        hidden_states = self.dense(hidden_states)

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class BertFrontDecomposition(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention

        weight = split_layer.intermediate.dense.weight
        u, s, v = torch.linalg.svd(weight)

        self.dense_s = nn.Linear(in_features=1, out_features=1, bias=False)
        self.dense_v = nn.Linear(in_features=1, out_features=1, bias=False)

        self.dense_s.weight = nn.Parameter(torch.diag(s[:rank]))
        self.dense_v.weight = nn.Parameter(v[:rank].clone())

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.attention(hidden_states, extended_attention_mask)[0]
        hidden_states = self.dense_v(hidden_states)
        hidden_states = self.dense_s(hidden_states)

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class BertFrontOriQuant(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num)])

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = torch.quantize_per_tensor(hidden_states, scale=1.0, zero_point=0, dtype=torch.qint8)

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class BertFrontFFNQuant(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention
        self.dense = split_layer.intermediate.dense

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.attention(hidden_states, extended_attention_mask)[0]
        hidden_states = self.dense(hidden_states)

        hidden_states = torch.quantize_per_tensor(hidden_states, scale=1.0, zero_point=0, dtype=torch.qint8)

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class BertFrontFFNQuantRes(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention
        self.dense = split_layer.intermediate.dense

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        attention_output = self.attention(hidden_states, extended_attention_mask)[0]
        ffn_output = self.dense(attention_output)
        
        # Quantize both FFN output and residual connection
        ffn_output_q = torch.quantize_per_tensor(ffn_output, scale=1.0, zero_point=0, dtype=torch.qint8)
        residual_q = torch.quantize_per_tensor(attention_output, scale=1.0, zero_point=0, dtype=torch.qint8)
        
        return (ffn_output_q.cpu(), residual_q.cpu())

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class DistBert(nn.Module):
    def __init__(self, devices, pretrain_dir, split_layer, model_type):
        super().__init__()
        front_classes = {
            1: BertFrontOri,
            2: BertFrontFFN,
            3: BertFrontDecomposition,
            4: BertFrontOriQuant,
            5: BertFrontFFNQuant,
            6: BertFrontFFNQuantRes
        }
        self.front_ref = rpc.remote(
            "cloud", 
            front_classes[model_type], 
            args=(pretrain_dir, split_layer, 8, devices["cloud"])  # rank=8 for decomposition
        )
        
        self.cloud_worker = rpc.remote("edge", CloudWorker)
        self.cloud_worker.rpc_sync().initialize_model(
            pretrain_dir,
            split_layer,
            model_type
        )
        print("Cloud worker and model initialised")
        
    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        
        #start_comm = time.perf_counter()
        #self.cloud_worker.rpc_sync().receive_tensor(hidden_states)
        #end_comm = time.perf_counter()
        #comm_time = (end_comm - start_comm) * 1000
        #print(f"RPC communication time: {comm_time:.2f} ms")
        
        # Continue with computation
        outputs = self.cloud_worker.rpc_sync().forward(hidden_states, attn_mask)
        return outputs
    
    def check_devices(self):
        """Print device information for both parts of the model"""
        print("\nDistributed Model Devices:")
        front_device = self.front_ref.rpc_sync().get_device()
        back_device = self.cloud_worker.rpc_sync().get_device()
        print(f"Front model (edge) device: {front_device}")
        print(f"Back model (cloud) device: {back_device}")
        
    def load_state_dict(self, front_state_dict, back_state_dict):
        self.front_ref.rpc_sync().load_state_dict(front_state_dict)
        self.cloud_worker.rpc_sync().load_state_dict(back_state_dict)

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.cloud_worker.rpc_sync().parameter_rrefs())  # Changed from back_ref
        return remote_params