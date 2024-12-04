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
from model_classes import MidiBertBack, MidiBertBackFFN, MidiBertBackFFNDecomp, MidiBertBackQuant, MidiBertBackFFNQuant, MidiBertBackFFNQuantRes, CloudWorker

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class MidiBertFront(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        self.split_layer = split_layer
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.hidden_size)

        #self.embedding_dropout = nn.Dropout(p=0.5)
        #self.classification_dropout = nn.Dropout(p=0.5)

        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] for i in range(split_layer) # Only keep layers up to split_layer
        ])

    def forward(self, input_ids, attn_mask=None):
        input_ids = input_ids.to('cpu')
        if attn_mask is not None:
            attn_mask = attn_mask.to('cpu')
    
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        #emb_linear = self.embedding_dropout(emb_linear)
        
        outputs = self.bert(
            inputs_embeds=emb_linear,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        #output = self.classification_dropout(outputs.last_hidden_state)

        return outputs.last_hidden_state.cpu()
        #return output.cpu()
        #return outputs.hidden_states[self.split_layer]

    def get_device(self):
        # Return device of first parameter in the model
        return next(self.parameters()).device

    def parameter_rrefs(self):
        """Create RRefs for parameters for distributed training"""
        return [rpc.RRef(param) for param in self.parameters()]
    
class MidiBertBack2(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print("\nCUDA Debug (MidiBertBack init):")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Selected device: {self.device}")
        
        self.bert = BertModel(bertConfig)

        self.bert.embeddings = None
        
        # Keep only the layers from split_layer onwards
        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] 
            for i in range(split_layer, len(original_layers))
        ])
        
        self.to(self.device)
        
        print("\nModel device verification:")
        print(f"Model device: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} device: {next(layer.parameters()).device}")

    def forward(self, hidden_states, attn_mask=None):
    
        hidden_states = hidden_states.to(self.device, non_blocking=True)

        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to(self.device)
        else:
            extended_attention_mask = None
            
        sequence_output = hidden_states
        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                sequence_output,
                attention_mask=extended_attention_mask
            )
            sequence_output = layer_outputs[0]
            
        return BaseModelOutput(
            last_hidden_state=sequence_output,#.cpu(),
            hidden_states=None
        )
    
    def get_device(self):
        # Return device of first parameter in the model
        return next(self.parameters()).device

    def parameter_rrefs(self):
        """Create RRefs for parameters for distributed training"""
        return [rpc.RRef(param) for param in self.parameters()]
    
class DistMidiBert(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer, model_type):
        super().__init__()
        front_classes = {
            1: MidiBertFront,
            2: MidiBertFrontFFN,
            3: MidiBertFrontFFNDecomp,
            4: MidiBertFrontQuant,
            5: MidiBertFrontFFNQuant,
            6: MidiBertFrontFFNQuantRes,
            
        }
        self.front_ref = rpc.remote(
            "edge", 
            front_classes[model_type], 
            args=(bertConfig, e2w, w2e, split_layer)
        )
        
        self.cloud_worker = rpc.remote("cloud", CloudWorker)
        self.cloud_worker.rpc_sync().initialize_model(
            bertConfig, 
            split_layer,
            model_type
        )
        print("Cloud worker and model initialized")
    
    '''def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.cloud_worker.rpc_sync().forward(hidden_states, attn_mask)
        return outputs'''
        
    def forward2(self, input_ids, attn_mask=None):
        
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        
        # Forward through back model
        outputs = self.cloud_worker.rpc_sync().forward(hidden_states, attn_mask)
        end_comm = time.perf_counter()
        comm_time = (end_comm - start_comm) * 1000
        print(f"Total RPC communication time: {comm_time:.2f} ms")
        return outputs
        
    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        
        # Measure only RPC send time
        send_start = time.perf_counter()
        self.cloud_worker.rpc_sync().receive_tensor(hidden_states)
        send_end = time.perf_counter()
        send_time = (send_end - send_start) * 1000
        print(f"RPC send time: {send_time:.2f} ms")
        
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
    
def load_split_model_from_checkpoint(checkpoint_path, bertConfig, e2w, w2e, split_layer):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    front_model = MidiBertFront(bertConfig, e2w, w2e, split_layer)
    #back_model = MidiBertBack(bertConfig, split_layer)

    #front_state_dict = {}
    back_state_dict = {}
    '''
    for key, value in state_dict.items():
        if 'bert.encoder.layer.' in key:
            layer_idx = int(key.split('.')[3])
            if layer_idx < split_layer:
                front_state_dict[key] = value
            else:
                new_key = key.replace(
                    f'bert.encoder.layer.{layer_idx}',
                    f'bert.encoder.layer.{layer_idx - split_layer}'
                )
                back_state_dict[new_key] = value
        elif any(x in key for x in ['word_emb', 'in_linear', 'bert.embeddings']):
            front_state_dict[key] = value
        elif 'bert.pooler' in key:
            back_state_dict[key] = value'''
            
    for key, value in state_dict.items():
        if 'bert.encoder.layer.' in key:
            layer_idx = int(key.split('.')[3])
            if layer_idx >= split_layer:
                new_key = key.replace(
                    f'bert.encoder.layer.{layer_idx}',
                    f'bert.encoder.layer.{layer_idx - split_layer}'
                )
                back_state_dict[new_key] = value
        elif 'bert.pooler' in key:
            back_state_dict[key] = value
    
    # Return front model and state dict for back model
    return front_model, back_state_dict
    '''
    def count_parameters(state_dict):
        return sum(p.numel() for p in state_dict.values())
    
    front_params = count_parameters(front_state_dict)
    back_params = count_parameters(back_state_dict)
    
    missing_front = front_model.load_state_dict(front_state_dict, strict=False)
    missing_back = back_model.load_state_dict(back_state_dict, strict=False)
    
    print("\nFront Model Loading Details:")
    print(f"Number of parameter tensors: {len(front_state_dict)}")
    print(f"Total parameters loaded: {front_params:,}")
    print(f"Missing keys ({len(missing_front.missing_keys)}):")
    for key in missing_front.missing_keys:
        print(f"  {key}")
    
    print("\nBack Model Loading Details:")
    print(f"Number of parameter tensors: {len(back_state_dict)}")
    print(f"Total parameters loaded: {back_params:,}")
    print(f"Missing keys ({len(missing_back.missing_keys)}):")
    for key in missing_back.missing_keys:
        print(f"  {key}")
    
    print(f"\nTotal parameters across both models: {front_params + back_params:,}")
    
    return front_model, back_model
'''
class MidiBertFrontFFN(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        self.split_layer = split_layer
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.hidden_size)

        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] for i in range(split_layer)
        ])

        split_transformer = original_layers[split_layer]
        self.split_attention = split_transformer.attention
        self.split_intermediate = split_transformer.intermediate.dense
        
        #self.attention_layernorm = nn.LayerNorm(bertConfig.hidden_size)

    def forward(self, input_ids, attn_mask=None):
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        hidden_states = self.in_linear(embs)
        
        if attn_mask is not None:
            attention_mask = attn_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        attention_output = self.split_attention(hidden_states, attention_mask)[0]
        intermediate_output = self.split_intermediate(attention_output)
        
        return intermediate_output.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]

class MidiBertBackFFN2(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] 
            for i in range(split_layer + 1, len(original_layers))
        ])

        self.to("cuda:0")

    def forward(self, hidden_states, attn_mask=None):
        hidden_states = hidden_states.to("cuda:0")

        if attn_mask is not None:
            attn_mask = attn_mask.to("cuda:0")
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to("cuda:0")
        else:
            attention_mask = None
            
        sequence_output = self.intermediate_act_fn(hidden_states)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                sequence_output,
                attention_mask=extended_attention_mask
            )
            sequence_output = layer_outputs[0]
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
    
class DistMidiBertFFN(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        
        self.front_ref = rpc.remote("edge", MidiBertFrontFFN, args=(bertConfig, e2w, w2e, split_layer))
        self.back_ref = rpc.remote("cloud", MidiBertBackFFN, args=(bertConfig, split_layer))

    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.back_ref.rpc_sync().forward(hidden_states, attn_mask)
        return outputs

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())
        return remote_params

def load_split_model_from_checkpoint_FFN(checkpoint_path, bertConfig, e2w, w2e, split_layer):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    front_model = MidiBertFrontFFN(bertConfig, e2w, w2e, split_layer)
    front_state_dict = {}
    back_state_dict = {}
    
    for key, value in state_dict.items():
        if 'bert.encoder.layer.' in key:
            layer_idx = int(key.split('.')[3])
            if layer_idx < split_layer:
                front_state_dict[key] = value
            elif layer_idx == split_layer:
                if 'attention' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.attention', 'split_attention')
                    front_state_dict[new_key] = value
                elif 'intermediate.dense' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.intermediate.dense', 'split_intermediate')
                    front_state_dict[new_key] = value
                elif 'output' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.output', 'split_output')
                    back_state_dict[new_key] = value
            else:
                new_key = key.replace(
                    f'bert.encoder.layer.{layer_idx}',
                    f'bert.encoder.layer.{layer_idx - split_layer - 1}'
                )
                back_state_dict[new_key] = value
        elif any(x in key for x in ['word_emb', 'in_linear', 'bert.embeddings']):
            front_state_dict[key] = value
        elif 'bert.pooler' in key:
            back_state_dict[key] = value
            
    if f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight' in state_dict:
        front_state_dict['attention_layernorm.weight'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight']
        front_state_dict['attention_layernorm.bias'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.bias']

    missing_front = front_model.load_state_dict(front_state_dict, strict=False)
    
    print("\nFront Model Loading Details:")
    print(f"Number of parameter tensors: {len(front_state_dict)}")
    print(f"Total parameters loaded: {sum(p.numel() for p in front_state_dict.values()):,}")
    print(f"Missing keys ({len(missing_front.missing_keys)}):")
    for key in missing_front.missing_keys:
        print(f"  {key}")
    
    return front_model, back_state_dict
    
def load_split_model_from_checkpoint_FFN2(checkpoint_path, bertConfig, e2w, w2e, split_layer):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    front_model = MidiBertFrontFFN(bertConfig, e2w, w2e, split_layer)
    back_model = MidiBertBackFFN(bertConfig, split_layer)
    
    front_state_dict = {}
    back_state_dict = {}
    
    for key, value in state_dict.items():
        if 'bert.encoder.layer.' in key:
            layer_idx = int(key.split('.')[3])
            if layer_idx < split_layer:
                front_state_dict[key] = value
            elif layer_idx == split_layer:
                if 'attention' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.attention', 'split_attention')
                    front_state_dict[new_key] = value
                elif 'intermediate.dense' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.intermediate.dense', 'split_intermediate')
                    front_state_dict[new_key] = value
                elif 'output' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.output', 'split_output')
                    back_state_dict[new_key] = value
            else:
                new_key = key.replace(
                    f'bert.encoder.layer.{layer_idx}',
                    f'bert.encoder.layer.{layer_idx - split_layer - 1}'
                )
                back_state_dict[new_key] = value
        elif any(x in key for x in ['word_emb', 'in_linear', 'bert.embeddings']):
            front_state_dict[key] = value
        elif 'bert.pooler' in key:
            back_state_dict[key] = value
            
    if f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight' in state_dict:
        front_state_dict['attention_layernorm.weight'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight']
        front_state_dict['attention_layernorm.bias'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.bias']

    # Print state dict keys for debugging
    #print("\nFront state dict keys:")
    #for key in sorted(front_state_dict.keys()):
    #    print(f"  {key}")
    
    #print("\nBack state dict keys:")
    #for key in sorted(back_state_dict.keys()):
    #    print(f"  {key}")
            
    def count_parameters(state_dict):
        return sum(p.numel() for p in state_dict.values())
    
    front_params = count_parameters(front_state_dict)
    back_params = count_parameters(back_state_dict)
    
    missing_front = front_model.load_state_dict(front_state_dict, strict=False)
    missing_back = back_model.load_state_dict(back_state_dict, strict=False)
    
    print("\nFront Model Loading Details:")
    print(f"Number of parameter tensors: {len(front_state_dict)}")
    print(f"Total parameters loaded: {front_params:,}")
    print(f"Missing keys ({len(missing_front.missing_keys)}):")
    for key in missing_front.missing_keys:
        print(f"  {key}")
    
    print("\nBack Model Loading Details:")
    print(f"Number of parameter tensors: {len(back_state_dict)}")
    print(f"Total parameters loaded: {back_params:,}")
    print(f"Missing keys ({len(missing_back.missing_keys)}):")
    for key in missing_back.missing_keys:
        print(f"  {key}")
    
    print(f"\nTotal parameters across both models: {front_params + back_params:,}")
    
    return front_model, back_model

class MidiBertFrontFFNDecomp(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        self.split_layer = split_layer
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.hidden_size)

        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] for i in range(split_layer)
        ])

        split_transformer = original_layers[split_layer]
        self.split_attention = split_transformer.attention

        weight = split_transformer.intermediate.dense.weight
        u, s, v = torch.linalg.svd(weight)

        self.dense_s = nn.Linear(in_features=1, out_features=1, bias=False)
        self.dense_v = nn.Linear(in_features=1, out_features=1, bias=False)

        self.dense_s.weight = nn.Parameter(torch.diag(s[:8])) #rank = 8
        self.dense_v.weight = nn.Parameter(v[:8].clone()) #rank = 8
        
        self.attention_layernorm = nn.LayerNorm(bertConfig.hidden_size)

    def forward(self, input_ids, attn_mask=None):
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        hidden_states = self.in_linear(embs)
        
        if attn_mask is not None:
            attention_mask = attn_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        attention_output = self.split_attention(hidden_states, attention_mask)[0]
        #attention_output = self.attention_layernorm(attention_output + hidden_states)
        intermediate_output = self.dense_v(attention_output)
        intermediate_output = self.dense_s(intermediate_output)
        
        return intermediate_output.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]

class MidiBertBackFFNDecomp2(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output

        weight = split_transformer.intermediate.dense.weight
        bias = split_transformer.intermediate.dense.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=1, out_features=1, bias=True)

        self.dense_u.weight = nn.Parameter(u[:, :8].clone()) #rank = 8
        self.dense_u.bias = bias
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] 
            for i in range(split_layer + 1, len(original_layers))
        ])

        self.to("cuda:0")

    def forward(self, hidden_states, attn_mask=None):
        hidden_states = hidden_states.to("cuda:0")

        if attn_mask is not None:
            attn_mask = attn_mask.to("cuda:0")
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to("cuda:0")
        else:
            attention_mask = None
            
        sequence_output = self.dense_u(hidden_states)
        sequence_output = self.intermediate_act_fn(sequence_output)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                sequence_output,
                attention_mask=extended_attention_mask
            )
            sequence_output = layer_outputs[0]
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
    
class DistMidiBertFFNDecomp(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        
        self.front_ref = rpc.remote("edge", MidiBertFrontFFNDecomp, args=(bertConfig, e2w, w2e, split_layer))
        self.back_ref = rpc.remote("cloud", MidiBertBackFFNDecomp, args=(bertConfig, split_layer))

    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.back_ref.rpc_sync().forward(hidden_states, attn_mask)
        return outputs

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())
        return remote_params
        
def load_split_model_from_checkpoint_FFN_Decomp(checkpoint_path, bertConfig, e2w, w2e, split_layer):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    front_model = MidiBertFrontFFNDecomp(bertConfig, e2w, w2e, split_layer)
    front_state_dict = {}
    back_state_dict = {}
    
    # Get the weight for SVD decomposition
    split_layer_weight = None
    split_layer_bias = None
    for key, value in state_dict.items():
        if f'bert.encoder.layer.{split_layer}.intermediate.dense.weight' in key:
            split_layer_weight = value
        elif f'bert.encoder.layer.{split_layer}.intermediate.dense.bias' in key:
            split_layer_bias = value
    
    # Perform SVD and add decomposed weights to back_state_dict
    if split_layer_weight is not None:
        u, s, v = torch.linalg.svd(split_layer_weight)
        back_state_dict['dense_u.weight'] = u[:, :8].clone()  # rank = 8
        back_state_dict['dense_u.bias'] = split_layer_bias if split_layer_bias is not None else None
    
    # Rest of state dict loading
    for key, value in state_dict.items():
        if 'bert.encoder.layer.' in key:
            layer_idx = int(key.split('.')[3])
            if layer_idx < split_layer:
                front_state_dict[key] = value
            elif layer_idx == split_layer:
                if 'attention' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.attention', 'split_attention')
                    front_state_dict[new_key] = value
                elif 'intermediate.dense' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.intermediate.dense', 'split_intermediate')
                    front_state_dict[new_key] = value
                elif 'output' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.output', 'split_output')
                    back_state_dict[new_key] = value
            else:
                new_key = key.replace(
                    f'bert.encoder.layer.{layer_idx}',
                    f'bert.encoder.layer.{layer_idx - split_layer - 1}'
                )
                back_state_dict[new_key] = value
        elif any(x in key for x in ['word_emb', 'in_linear', 'bert.embeddings']):
            front_state_dict[key] = value
        elif 'bert.pooler' in key:
            back_state_dict[key] = value
            
    if f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight' in state_dict:
        front_state_dict['attention_layernorm.weight'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight']
        front_state_dict['attention_layernorm.bias'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.bias']

    missing_front = front_model.load_state_dict(front_state_dict, strict=False)
    
    print("\nFront Model Loading Details:")
    print(f"Number of parameter tensors: {len(front_state_dict)}")
    print(f"Total parameters loaded: {sum(p.numel() for p in front_state_dict.values()):,}")
    print(f"Missing keys ({len(missing_front.missing_keys)}):")
    for key in missing_front.missing_keys:
        print(f"  {key}")
    
    return front_model, back_state_dict

def load_split_model_from_checkpoint_FFN_Decomp2(checkpoint_path, bertConfig, e2w, w2e, split_layer):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    front_model = MidiBertFrontFFNDecomp(bertConfig, e2w, w2e, split_layer)
    back_model = MidiBertBackFFNDecomp(bertConfig, split_layer)
    
    front_state_dict = {}
    back_state_dict = {}
    
    for key, value in state_dict.items():
        if 'bert.encoder.layer.' in key:
            layer_idx = int(key.split('.')[3])
            if layer_idx < split_layer:
                front_state_dict[key] = value
            elif layer_idx == split_layer:
                if 'attention' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.attention', 'split_attention')
                    front_state_dict[new_key] = value
                elif 'intermediate.dense' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.intermediate.dense', 'split_intermediate')
                    front_state_dict[new_key] = value
                elif 'output' in key:
                    new_key = key.replace(f'bert.encoder.layer.{layer_idx}.output', 'split_output')
                    back_state_dict[new_key] = value
            else:
                new_key = key.replace(
                    f'bert.encoder.layer.{layer_idx}',
                    f'bert.encoder.layer.{layer_idx - split_layer - 1}'
                )
                back_state_dict[new_key] = value
        elif any(x in key for x in ['word_emb', 'in_linear', 'bert.embeddings']):
            front_state_dict[key] = value
        elif 'bert.pooler' in key:
            back_state_dict[key] = value
            
    if f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight' in state_dict:
        front_state_dict['attention_layernorm.weight'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.weight']
        front_state_dict['attention_layernorm.bias'] = state_dict[f'bert.encoder.layer.{split_layer}.attention.output.LayerNorm.bias']

    # Print state dict keys for debugging
    #print("\nFront state dict keys:")
    #for key in sorted(front_state_dict.keys()):
    #    print(f"  {key}")
    
    #print("\nBack state dict keys:")
    #for key in sorted(back_state_dict.keys()):
    #    print(f"  {key}")
            
    def count_parameters(state_dict):
        return sum(p.numel() for p in state_dict.values())
    
    front_params = count_parameters(front_state_dict)
    back_params = count_parameters(back_state_dict)
    
    missing_front = front_model.load_state_dict(front_state_dict, strict=False)
    missing_back = back_model.load_state_dict(back_state_dict, strict=False)
    
    print("\nFront Model Loading Details:")
    print(f"Number of parameter tensors: {len(front_state_dict)}")
    print(f"Total parameters loaded: {front_params:,}")
    print(f"Missing keys ({len(missing_front.missing_keys)}):")
    for key in missing_front.missing_keys:
        print(f"  {key}")
    
    print("\nBack Model Loading Details:")
    print(f"Number of parameter tensors: {len(back_state_dict)}")
    print(f"Total parameters loaded: {back_params:,}")
    print(f"Missing keys ({len(missing_back.missing_keys)}):")
    for key in missing_back.missing_keys:
        print(f"  {key}")
    
    print(f"\nTotal parameters across both models: {front_params + back_params:,}")
    
    return front_model, back_model

class MidiBertFrontQuant(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        self.split_layer = split_layer
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.hidden_size)

        #self.embedding_dropout = nn.Dropout(p=0.5)
        #self.classification_dropout = nn.Dropout(p=0.5)

        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] for i in range(split_layer) # Only keep layers up to split_layer
        ])

    def forward(self, input_ids, attn_mask=None):
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)
        
        outputs = self.bert(
            inputs_embeds=emb_linear,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.last_hidden_state
        hidden_states_quantized = torch.quantize_per_tensor(
            hidden_states,
            scale=1.0,
            zero_point=0,
            dtype=torch.qint8
        )
        
        return hidden_states_quantized.cpu()

    def get_device(self):
        # Return device of first parameter in the model
        return next(self.parameters()).device

    def parameter_rrefs(self):
        """Create RRefs for parameters for distributed training"""
        return [rpc.RRef(param) for param in self.parameters()]
    
class MidiBertBackQuant2(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        
        self.bert = BertModel(bertConfig)

        self.bert.embeddings = None
        
        # Keep only the layers from split_layer onwards
        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] 
            for i in range(split_layer, len(original_layers))
        ])

        self.to("cuda:0")

    def forward(self, hidden_states_quantized, attn_mask=None):
        hidden_states = torch.dequantize(hidden_states_quantized).to("cuda:0")

        if attn_mask is not None:
            attn_mask = attn_mask.to("cuda:0")
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to("cuda:0")
        else:
            extended_attention_mask = None
            
        sequence_output = hidden_states
        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                sequence_output,
                attention_mask=extended_attention_mask
            )
            sequence_output = layer_outputs[0]
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )
    
    def get_device(self):
        # Return device of first parameter in the model
        return next(self.parameters()).device

    def parameter_rrefs(self):
        """Create RRefs for parameters for distributed training"""
        return [rpc.RRef(param) for param in self.parameters()]
        
class DistMidiBertQuant(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        
        self.front_ref = rpc.remote(
            "edge", 
            MidiBertFront, 
            args=(bertConfig, e2w, w2e, split_layer)
        )
        
        self.cloud_worker = rpc.remote("cloud", CloudWorker)
        self.cloud_worker.rpc_sync().initialize_model(bertConfig, split_layer)
        print("Cloud worker and model initialized")
    
    '''def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.cloud_worker.rpc_sync().forward(hidden_states, attn_mask)
        return outputs'''
        
    def forward(self, input_ids, attn_mask=None):
        
        # Forward through front model
        start_comm = time.perf_counter()
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        
        # Forward through back model
        outputs = self.cloud_worker.rpc_sync().forward(hidden_states, attn_mask)
        end_comm = time.perf_counter()
        comm_time = (end_comm - start_comm) * 1000
        print(f"Total RPC communication time: {comm_time:.2f} ms")
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
    
class DistMidiBertQuant2(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        
        self.front_ref = rpc.remote(
            "edge", 
            MidiBertFrontQuant,
            args=(bertConfig, e2w, w2e, split_layer)
        )
        
        self.back_ref = rpc.remote(
            "cloud", 
            MidiBertBackQuant,
            args=(bertConfig, split_layer)
        )
    
    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.back_ref.rpc_sync().forward(hidden_states, attn_mask)
        return outputs
    
    def check_devices(self):
        """Print device information for both parts of the model"""
        print("\nDistributed Model Devices:")
        front_device = self.front_ref.rpc_sync().get_device()
        back_device = self.back_ref.rpc_sync().get_device()
        print(f"Front model (edge) device: {front_device}")
        print(f"Back model (cloud) device: {back_device}")

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())
        return remote_params
    
class MidiBertFrontFFNQuant(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        self.split_layer = split_layer
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.hidden_size)

        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] for i in range(split_layer)
        ])

        split_transformer = original_layers[split_layer]
        self.split_attention = split_transformer.attention
        self.split_intermediate = split_transformer.intermediate.dense
        
        #self.attention_layernorm = nn.LayerNorm(bertConfig.hidden_size)

    def forward(self, input_ids, attn_mask=None):
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        hidden_states = self.in_linear(embs)
        
        if attn_mask is not None:
            attention_mask = attn_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        attention_output = self.split_attention(hidden_states, attention_mask)[0]
        intermediate_output = self.split_intermediate(attention_output)
        
        quantized_output = torch.quantize_per_tensor(
            intermediate_output,
            scale=1.0,
            zero_point=0,
            dtype=torch.qint8
        )
        
        return quantized_output.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]

class MidiBertBackFFNQuant2(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] 
            for i in range(split_layer + 1, len(original_layers))
        ])

        self.to("cuda:0")

    def forward(self, hidden_states, attn_mask=None):
        hidden_states = torch.dequantize(hidden_states).to("cuda:0")

        if attn_mask is not None:
            attn_mask = attn_mask.to("cuda:0")
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to("cuda:0")
        else:
            attention_mask = None
            
        sequence_output = self.intermediate_act_fn(hidden_states)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                sequence_output,
                attention_mask=extended_attention_mask
            )
            sequence_output = layer_outputs[0]
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
    
class DistMidiBertFFNQuant(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        
        self.front_ref = rpc.remote("edge", MidiBertFrontFFNQuant, args=(bertConfig, e2w, w2e, split_layer))
        self.back_ref = rpc.remote("cloud", MidiBertBackFFNQuant, args=(bertConfig, split_layer))

    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.back_ref.rpc_sync().forward(hidden_states, attn_mask)
        return outputs

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())
        return remote_params

class MidiBertFrontFFNQuantRes(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        self.split_layer = split_layer
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.hidden_size)

        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] for i in range(split_layer)
        ])

        split_transformer = original_layers[split_layer]
        self.split_attention = split_transformer.attention
        self.split_intermediate = split_transformer.intermediate.dense
        
        #self.attention_layernorm = nn.LayerNorm(bertConfig.hidden_size)

    def forward(self, input_ids, attn_mask=None):
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        hidden_states = self.in_linear(embs)
        
        if attn_mask is not None:
            attention_mask = attn_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        attention_output = self.split_attention(hidden_states, attention_mask)[0]
        intermediate_output = self.split_intermediate(attention_output)
        
        quantized_ffn = torch.quantize_per_tensor(
            intermediate_output,
            scale=1.0,
            zero_point=0,
            dtype=torch.qint8
        )
        
        quantized_residual = torch.quantize_per_tensor(
            attention_output,
            scale=1.0,
            zero_point=0,
            dtype=torch.qint8
        )
        
        return (quantized_ffn.cpu(), quantized_residual.cpu())

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]

class MidiBertBackFFNQuantRes2(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i] 
            for i in range(split_layer + 1, len(original_layers))
        ])

        self.to("cuda:0")

    def forward(self, hidden_states_tuple, attn_mask=None):
        quantized_ffn, quantized_residual = hidden_states_tuple
        
        ffn_output = torch.dequantize(quantized_ffn).to("cuda:0")
        residual = torch.dequantize(quantized_residual).to("cuda:0")

        if attn_mask is not None:
            attn_mask = attn_mask.to("cuda:0")
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to("cuda:0")
        else:
            attention_mask = None
            
        sequence_output = self.intermediate_act_fn(ffn_output)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        
        sequence_output = sequence_output + residual
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for layer in self.bert.encoder.layer:
            layer_outputs = layer(
                sequence_output,
                attention_mask=extended_attention_mask
            )
            sequence_output = layer_outputs[0]
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
    
class DistMidiBertFFNQuantRes(nn.Module):
    def __init__(self, devices, bertConfig, e2w, w2e, split_layer):
        super().__init__()
        
        self.front_ref = rpc.remote("edge", MidiBertFrontFFNQuantRes, args=(bertConfig, e2w, w2e, split_layer))
        self.back_ref = rpc.remote("cloud", MidiBertBackFFNQuantRes, args=(bertConfig, split_layer))

    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attn_mask)
        outputs = self.back_ref.rpc_sync().forward(hidden_states, attn_mask)
        return outputs

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())
        return remote_params

if __name__ == "__main__":
    import pickle
    from transformers import BertConfig
    from data import prepare_data
    
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    np.random.seed(2021)
    random.seed(2021)
    
    dict_path = './data/dict/CP.pkl'
    print("\nLoading Dictionary...")
    with open(dict_path, 'rb') as f:
        e2w, w2e = pickle.load(f)
    
    config = BertConfig(
        max_position_embeddings=512,
        position_embedding_type='relative_key_query',
        hidden_size=768
    )
    
    checkpoint_path = './pretrain/pretrain_model.ckpt'
    print("\nCreating split models...")
    front_model, back_model = load_split_model_from_checkpoint_FFN_Decomp(
        checkpoint_path, config, e2w, w2e, split_layer=6
    )
    
    print("\nLoading data...")
    class Args:
        def __init__(self):
            self.dataset = 'composer'
            self.task = 'composer'
            self.batch_size = 12
            self.num_workers = 5
            self.data_root = './data/dict/CP_data'
            self.max_seq_len = 512
            self.hidden_size = 768
    
    args = Args()
    data_loaders, data_info = prepare_data(args)
    train_loader, valid_loader, test_loader = data_loaders
    
    batch_data, batch_labels = next(iter(train_loader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    attn_mask = torch.ones((batch_data.shape[0], batch_data.shape[1]))
    
    print("\nTesting forward passes...")
    try:
        hidden_states = front_model(batch_data, attn_mask)
        print("Front model output shape:", hidden_states.shape)
        
        outputs = back_model(hidden_states, attn_mask)
        print("Back model output shapes:")
        print("- Last hidden state:", outputs.last_hidden_state.shape)
        
        print("\nForward passes successful!")
        
    except Exception as e:
        print(f"Error during forward passes: {e}")