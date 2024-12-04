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

class MidiBertBack(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print(f"MidiBertBack initializing explicitly on cuda:0")
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        # Keep only the layers from split_layer onwards and move each to cuda:0
        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i].to("cuda:0") 
            for i in range(split_layer, len(original_layers))
        ])
        
        # Explicitly move entire model to cuda:0
        self.to("cuda:0")
        
        print("\nModel device verification:")
        print(f"Model is on: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} is on: {next(layer.parameters()).device}")

    def forward(self, hidden_states, attn_mask=None):
        print(f"\nMidiBertBack Forward:")
        print(f"Input tensor device: {hidden_states.device}")
        print(f"Model device: {next(self.parameters()).device}")
        
        # Explicitly move to cuda:0
        hidden_states = hidden_states.to("cuda:0")
        
        sequence_output = hidden_states
        for i, layer in enumerate(self.bert.encoder.layer):
            sequence_output = sequence_output.to("cuda:0")
            layer_outputs = layer(sequence_output, attention_mask=attn_mask)
            sequence_output = layer_outputs[0]
            print(f"Layer {i} output device: {sequence_output.device}")
        
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
        
class MidiBertBackFFN(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print(f"MidiBertBackFFN initializing explicitly on cuda:0")
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i].to("cuda:0") 
            for i in range(split_layer + 1, len(original_layers))
        ])
        
        self.to("cuda:0")
        print("\nModel device verification:")
        print(f"Model is on: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} is on: {next(layer.parameters()).device}")

    def forward(self, hidden_states, attn_mask=None):
        print(f"\nMidiBertBackFFN Forward:")
        print(f"Input tensor device: {hidden_states.device}")
        print(f"Model device: {next(self.parameters()).device}")
        
        hidden_states = hidden_states.to("cuda:0")
            
        sequence_output = self.intermediate_act_fn(hidden_states)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for i, layer in enumerate(self.bert.encoder.layer):
            sequence_output = sequence_output.to("cuda:0")
            layer_outputs = layer(sequence_output, attention_mask=attn_mask)
            sequence_output = layer_outputs[0]
            print(f"Layer {i} output device: {sequence_output.device}")
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def get_device(self):
        return next(self.parameters()).device

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
        
class MidiBertBackFFNDecomp(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print(f"MidiBertBackFFNDecomp initializing explicitly on cuda:0")
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output

        # SVD decomposition
        weight = split_transformer.intermediate.dense.weight
        bias = split_transformer.intermediate.dense.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=1, out_features=1, bias=True)
        self.dense_u.weight = nn.Parameter(u[:, :8].clone()) #rank = 8
        self.dense_u.bias = bias
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i].to("cuda:0") 
            for i in range(split_layer + 1, len(original_layers))
        ])
        
        self.to("cuda:0")
        print("\nModel device verification:")
        print(f"Model is on: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} is on: {next(layer.parameters()).device}")

    def forward(self, hidden_states, attn_mask=None):
        print(f"\nMidiBertBackFFNDecomp Forward:")
        print(f"Input tensor device: {hidden_states.device}")
        print(f"Model device: {next(self.parameters()).device}")
        
        hidden_states = hidden_states.to("cuda:0")
            
        sequence_output = self.dense_u(hidden_states)
        sequence_output = self.intermediate_act_fn(sequence_output)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for i, layer in enumerate(self.bert.encoder.layer):
            sequence_output = sequence_output.to("cuda:0")
            layer_outputs = layer(sequence_output, attention_mask=attn_mask)
            sequence_output = layer_outputs[0]
            print(f"Layer {i} output device: {sequence_output.device}")
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def get_device(self):
        return next(self.parameters()).device

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
        
class MidiBertBackQuant(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print(f"MidiBertBackQuant initializing explicitly on cuda:0")
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i].to("cuda:0") 
            for i in range(split_layer, len(original_layers))
        ])
        
        self.to("cuda:0")
        print("\nModel device verification:")
        print(f"Model is on: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} is on: {next(layer.parameters()).device}")

    def forward(self, hidden_states_quantized, attn_mask=None):
        print(f"\nMidiBertBackQuant Forward:")
        print(f"Input tensor device: {hidden_states_quantized.device}")
        print(f"Model device: {next(self.parameters()).device}")
        
        hidden_states = torch.dequantize(hidden_states_quantized).to("cuda:0")
            
        sequence_output = hidden_states
        for i, layer in enumerate(self.bert.encoder.layer):
            sequence_output = sequence_output.to("cuda:0")
            layer_outputs = layer(sequence_output, attention_mask=attn_mask)
            sequence_output = layer_outputs[0]
            print(f"Layer {i} output device: {sequence_output.device}")
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def get_device(self):
        return next(self.parameters()).device

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
        
class MidiBertBackFFNQuant(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print(f"MidiBertBackFFNQuant initializing explicitly on cuda:0")
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i].to("cuda:0") 
            for i in range(split_layer + 1, len(original_layers))
        ])
        
        self.to("cuda:0")
        print("\nModel device verification:")
        print(f"Model is on: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} is on: {next(layer.parameters()).device}")

    def forward(self, hidden_states, attn_mask=None):
        print(f"\nMidiBertBackFFNQuant Forward:")
        print(f"Input tensor device: {hidden_states.device}")
        print(f"Model device: {next(self.parameters()).device}")
        
        hidden_states = torch.dequantize(hidden_states).to("cuda:0")
            
        sequence_output = self.intermediate_act_fn(hidden_states)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for i, layer in enumerate(self.bert.encoder.layer):
            sequence_output = sequence_output.to("cuda:0")
            layer_outputs = layer(sequence_output, attention_mask=attn_mask)
            sequence_output = layer_outputs[0]
            print(f"Layer {i} output device: {sequence_output.device}")
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def get_device(self):
        return next(self.parameters()).device

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
        
class MidiBertBackFFNQuantRes(nn.Module):
    def __init__(self, bertConfig, split_layer):
        super().__init__()
        print(f"MidiBertBackFFNQuantRes initializing explicitly on cuda:0")
        
        self.bert = BertModel(bertConfig)
        self.bert.embeddings = None
        
        original_layers = self.bert.encoder.layer
        split_transformer = original_layers[split_layer]
        self.intermediate_act_fn = split_transformer.intermediate.intermediate_act_fn
        self.split_output = split_transformer.output
        
        self.bert.encoder.layer = nn.ModuleList([
            original_layers[i].to("cuda:0") 
            for i in range(split_layer + 1, len(original_layers))
        ])
        
        self.to("cuda:0")
        print("\nModel device verification:")
        print(f"Model is on: {next(self.parameters()).device}")
        for i, layer in enumerate(self.bert.encoder.layer):
            print(f"Layer {i} is on: {next(layer.parameters()).device}")

    def forward(self, hidden_states_tuple, attn_mask=None):
        print(f"\nMidiBertBackFFNQuantRes Forward:")
        print(f"Model device: {next(self.parameters()).device}")
        
        quantized_ffn, quantized_residual = hidden_states_tuple
        ffn_output = torch.dequantize(quantized_ffn).to("cuda:0")
        residual = torch.dequantize(quantized_residual).to("cuda:0")
            
        sequence_output = self.intermediate_act_fn(ffn_output)
        sequence_output = self.split_output.dense(sequence_output)
        sequence_output = self.split_output.dropout(sequence_output)
        
        sequence_output = sequence_output + residual
        sequence_output = self.split_output.LayerNorm(sequence_output)

        for i, layer in enumerate(self.bert.encoder.layer):
            sequence_output = sequence_output.to("cuda:0")
            layer_outputs = layer(sequence_output, attention_mask=attn_mask)
            sequence_output = layer_outputs[0]
            print(f"Layer {i} output device: {sequence_output.device}")
            
        return BaseModelOutput(
            last_hidden_state=sequence_output.cpu(),
            hidden_states=None
        )

    def get_device(self):
        return next(self.parameters()).device

    def parameter_rrefs(self):
        return [rpc.RRef(param) for param in self.parameters()]
        
class CloudWorker:
    def __init__(self):
        self.model = None
        print("CloudWorker initialized")

    def initialize_model(self, bertConfig, split_layer, model_type):
        print(f"\nInitializing model type: {model_type}")
        model_classes = {
            1: MidiBertBack,
            2: MidiBertBackFFN,
            3: MidiBertBackFFNDecomp,
            4: MidiBertBackQuant,
            5: MidiBertBackFFNQuant,
            6: MidiBertBackFFNQuantRes
        }
        
        self.model = model_classes[model_type](bertConfig, split_layer)
        self.model.to("cuda:0")
        print(f"Model is on device: {next(self.model.parameters()).device}")
        return True

    def forward(self, hidden_states, attn_mask=None):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        print(f"\nCloud Worker Forward:")
        if isinstance(hidden_states, tuple):
            print(f"Input tensor devices: {hidden_states[0].device}, {hidden_states[1].device}")
        else:
            print(f"Input tensor device: {hidden_states.device}")
        
        start_receive = time.perf_counter()
        
        # Explicitly move to cuda:0
        if isinstance(hidden_states, tuple):
            hidden_states = (hidden_states[0].to("cuda:0"), hidden_states[1].to("cuda:0"))
            print(f"Hidden states moved to: cuda:0")
        else:
            hidden_states = hidden_states.to("cuda:0")
            print(f"Hidden states moved to: {hidden_states.device}")
        
        end_receive = time.perf_counter()
        receive_time = (end_receive - start_receive) * 1000
        print(f"Hidden states receive time: {receive_time:.2f} ms")
        
        if attn_mask is not None:
            attn_mask = attn_mask.to("cuda:0")
            extended_attention_mask = attn_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to("cuda:0")
        else:
            extended_attention_mask = None
        
        end_transfer = time.perf_counter()
        transfer_time = (end_transfer - start_transfer) * 1000
        print(f"Input transfer time: {transfer_time:.2f} ms")
    
        print(f"Model device before forward: {next(self.model.parameters()).device}")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Add profiling around the model forward pass
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            start_transfer = time.perf_counter()
            outputs = self.model(hidden_states, extended_attention_mask)
            end_transfer = time.perf_counter()
            transfer_time = (end_transfer - start_transfer) * 1000
        
        # Print profiling results
        print("\nGPU Operation Profiling (sorted by GPU total time):")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=30,
            max_src_column_width=100
        ))
        
        print("\nGPU Operation Profiling (sorted by CPU total time):")
        print(prof.key_averages().table(
            sort_by="cpu_time_total", 
            row_limit=30
        ))
        
        print(f"Output transfer time: {transfer_time:.2f} ms")
        
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"GPU Computation time: {elapsed_time:.2f} ms")
        
        return BaseModelOutput(
            last_hidden_state=outputs.last_hidden_state.cpu(),
            hidden_states=None
        )
        
    def parameter_rrefs(self):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return [rpc.RRef(p) for p in self.model.parameters()]

    def load_state_dict(self, state_dict):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.load_state_dict(state_dict)
        self.model.to("cuda:0")
        
        # Verify model is still on GPU after loading
        print("\nModel Device Verification after loading:")
        print(f"Model is on CUDA: {next(self.model.parameters()).is_cuda}")
        print(f"Current Memory Usage after loading:")
        print(torch.cuda.memory_summary())
        return True