import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import torch
import torch.distributed.rpc as rpc
from torch import nn
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.optim import AdamW
from tqdm import tqdm
import pickle
from datetime import datetime
import wandb
import random
import logging
import socket
import numpy as np

from transformers import BertConfig
import torch.profiler as profiler
from split_model2 import DistMidiBert, DistMidiBertFFN, DistMidiBertFFNDecomp, DistMidiBertQuant, DistMidiBertFFNQuant, DistMidiBertFFNQuantRes, load_split_model_from_checkpoint, load_split_model_from_checkpoint_FFN, load_split_model_from_checkpoint_FFN_Decomp
from data import prepare_data
import argparse
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description= 'Training for split MIDI-BERT')
    
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'],
                        required=True, default='composer', help='Fine-tuning task')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--split_layer', type=int, default=4)
    args = parser.parse_args()
    
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_rpc():
    os.environ.clear()
    
    os.environ['MASTER_ADDR'] = '192.168.42.236'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
    os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'

    logging.info("Initialising edge node...")
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=600,
        init_method=f'tcp://192.168.42.236:29500',
        _transports=['uv']
    )
    
    options.set_device_map("cloud", {"cpu": "cpu"})

    try:
        logging.info("Starting RPC initialisation...")
        rpc.init_rpc(
            "edge",
            rank=0,
            world_size=2,
            rpc_backend_options=options
        )
        logging.info("Edge RPC initialised successfully")
        return True
        
    except Exception as e:
        logging.error(f"RPC initialisation failed: {str(e)}")
        raise
        
def load_model_and_classifier_v2(model_choice, checkpoint_path, bertConfig, e2w, w2e, split_layer, devices):
    model_mapping = {
        1: {"name": "Split Learning (Original)", 
            "loader": load_split_model_from_checkpoint},
        2: {"name": "Split Fine-tuning", 
            "loader": load_split_model_from_checkpoint_FFN},
        3: {"name": "Split Fine-tuning with Decomposition", 
            "loader": load_split_model_from_checkpoint_FFN_Decomp},
        4: {"name": "Split Fine-tuning with Quantisation", 
            "loader": load_split_model_from_checkpoint},
        5: {"name": "Split Fine-tuning (FFN) with Quantisation", 
            "loader": load_split_model_from_checkpoint_FFN},
        6: {"name": "Split Fine-tuning (FFN) with Quantisation and Residual", 
            "loader": load_split_model_from_checkpoint_FFN}
    }

    print(f"Using {model_mapping[model_choice]['name']} model")
    
    front_model, back_model = model_mapping[model_choice]["loader"](
        checkpoint_path, 
        bertConfig, 
        e2w, 
        w2e, 
        split_layer
    )

    classifier = DistMidiBert(
        devices=devices,
        bertConfig=bertConfig,
        e2w=e2w,
        w2e=w2e,
        split_layer=split_layer,
        model_type=model_choice
    )

    return classifier, front_model, back_model

def save_split_checkpoint(model, optimizer, epoch, metrics, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        temp_file = filepath + '.tmp'

        if hasattr(optimizer, '_optim'):
            optimizer_state = optimizer._optim.state_dict()
        elif hasattr(optimizer, 'optim'):
            optimizer_state = optimizer.optim.state_dict()
        else:
            optimizer_state = None
            print("Warning: Could not save optimizer state - distributed optimizer state saving not supported")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            **metrics
        }
        
        torch.save(checkpoint, temp_file)
        
        os.replace(temp_file, filepath)
        print(f"Checkpoint saved successfully to {filepath}")
        
    except Exception as e:
        print(f"Warning: Could not save checkpoint to {filepath}")
        print(f"Error: {str(e)}")
        
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main():
    initialize_rpc()

    args = get_args()

    print("\nPlease select a model:")
    print("1: Split Fine-tuning (Original)")
    print("2: Split Fine-tuning (FFN)")
    print("3: Split Fine-tuning (FFN) with Decomposition")
    print("4: Split Fine-tuning (Original) with Quantisation")
    print("5: Split Fine-tuning (FFN) with Quantisation")
    print("6: Split Fine-tuning (FFN) with Quantisation and Residual")
    
    while True:
        try:
            model_choice = int(input("\nEnter your choice (1-6): "))
            if model_choice in [1, 2, 3, 4, 5, 6]:
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5 or 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    model_mapping = {
        1: {"name": "Split Learning", "group": "split_learning"},
        2: {"name": "Split Fine-tuning", "group": "split_finetuning"},
        3: {"name": "Split Fine-tuning with Decomposition", "group": "split_finetuning_decomp"},
        4: {"name": "Split Learning with Quantisation", "group": "split_finetuning_quant"},
        5: {"name": "Split Fine-tuning with Quantisation", "group": "split_finetuning_ffnquant"},
        6: {"name": "Split Fine-tuning with Quantisation and Residual", "group": "split_finetuning_ffnquantres"}
    }

    set_seed(2021)

    devices = {
        "edge": "cpu",
        "cloud": "cuda:0"
    }

    dict_path = './data/dict/CP.pkl'
    print("\nLoading dictionary...")
    with open(dict_path, 'rb') as f:
        e2w, w2e = pickle.load(f)

    bertConfig = BertConfig(
        max_position_embeddings=512,
        position_embedding_type='relative_key_query',
        hidden_size=768,
        attn_implementation="eager",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.5
    )

    checkpoint_path = './pretrain/pretrain_model.ckpt'
    split_layer = args.split_layer
    
    print("\nLoading split models from checkpoint...")
    classifier, front_model, back_state_dict = load_model_and_classifier_v2(
        model_choice, 
        checkpoint_path, 
        bertConfig, 
        e2w, 
        w2e, 
        split_layer,
        devices
    )
    
    classifier.load_state_dict(front_model.state_dict(), back_state_dict)

    class Args:
        def __init__(self):
            self.dataset = 'composer'
            self.task = 'composer'
            self.batch_size = 12
            self.num_workers = 5
            self.data_root = './data/dict/CP_data'

    args_data = Args()
    
    print("\nLoading data...")
    data_loaders, data_info = prepare_data(args_data)
    train_loader, valid_loader, test_loader = data_loaders

    criterion = nn.CrossEntropyLoss()
    
    opt = DistributedOptimizer(
        AdamW,
        classifier.parameter_rrefs(),
        lr=2e-5,
        weight_decay=0.01
    )

    print("\nStarting training...")

    best_valid_acc = 0.0
    patience_counter = 0
    patience_limit = 3  
    epochs = 10

    save_dir = os.path.join('./results', f"{args.task}_{args.name}")
    os.makedirs(save_dir, exist_ok=True)

    ### Training phase
    
    training_start_time = time.perf_counter()

    for epoch in range(epochs):
        classifier.train()

        total_train_loss = 0.0 
        correct_train_predictions_total = 0 
        total_train_samples=0

        trace_dir = f"./runs/{model_choice}_{args.split_layer}"
        os.makedirs(trace_dir, exist_ok=True)

        with tqdm(train_loader, desc=f"  Training") as train_tqdm:
            for i, batch in enumerate(train_tqdm):
                batch_input_ids = batch[0]
                batch_attention_mask = torch.ones(batch_input_ids.shape[:2])
                y_true = batch[1].to(devices["edge"])

                with dist_autograd.context() as context_id:
                    outputs = classifier(input_ids=batch_input_ids,
                                        attn_mask=batch_attention_mask)
                    y_pred = outputs.last_hidden_state.mean(dim=1)
                    loss = criterion(y_pred, y_true)

                    dist_autograd.backward(context_id, [loss])
                    
                    opt.step(context_id)
                
                    predictions = torch.argmax(y_pred, dim=-1)
                    correct = torch.eq(predictions, y_true).sum().item()
                    total = batch_input_ids.size(0)

                    total_train_loss += loss.item() * total
                    correct_train_predictions_total += correct
                    total_train_samples += total

                train_tqdm.set_postfix({"Loss": loss.item()})

        final_train_loss_total = (total_train_loss / total_train_samples)
        final_train_accuracy = (correct_train_predictions_total / total_train_samples)

        ### Validation phase

        classifier.eval()
        
        valid_loss_total = 0.0
        correct_val_predictions_total = 0
        
        with tqdm(valid_loader, desc=f"Validating") as valid_tqdm:
            for i, batch in enumerate(valid_tqdm):
                batch_input_ids = batch[0]
                batch_attention_mask = torch.ones(batch_input_ids.shape[:2])
                y_true = batch[1].to(devices["edge"])

                with torch.no_grad():
                    outputs = classifier(input_ids=batch_input_ids, attn_mask=batch_attention_mask)
                    y_pred = outputs.last_hidden_state.mean(dim=1).to(devices["edge"])
                    loss_val = criterion(y_pred, y_true).item()
                    valid_loss_total += loss_val * len(batch_input_ids)
                    
                    predictions_val = torch.argmax(y_pred, dim=-1)
                    correct = torch.eq(predictions_val, y_true).sum().item()
                    correct_val_predictions_total += correct

                valid_tqdm.set_postfix({"Loss": loss_val})

        final_valid_loss_total = valid_loss_total / len(valid_loader.dataset)
        final_valid_accuracy = (correct_val_predictions_total / len(valid_loader.dataset))
        
        clear_memory()

        ### Test phase

        test_loss_total = 0.0
        correct_test_predictions_total = 0
        
        with tqdm(test_loader, desc=f"   Testing") as test_tqdm:
            for i, batch in enumerate(test_tqdm):
                batch_input_ids = batch[0]
                batch_attention_mask = torch.ones(batch_input_ids.shape[:2])
                y_true = batch[1].to(devices["edge"])

                with torch.no_grad():
                    outputs = classifier(input_ids=batch_input_ids, attn_mask=batch_attention_mask)
                    y_pred = outputs.last_hidden_state.mean(dim=1).to(devices["edge"])
                    loss_test = criterion(y_pred, y_true).item()
                    test_loss_total += loss_test * len(batch_input_ids)
                    
                    predictions_test = torch.argmax(y_pred, dim=-1)
                    correct_test_predictions_total += torch.eq(predictions_test, y_true).sum().item()

                test_tqdm.set_postfix({"Loss": loss_test})

        final_test_loss_total = test_loss_total / len(test_loader.dataset)
        final_test_accuracy = (correct_test_predictions_total / len(test_loader.dataset))
        
        clear_memory()

        print(f"Epoch {epoch+1} completed.")
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training loss: {final_train_loss_total:.4f} |   Training accuracy: {final_train_accuracy:.4f}")
        print(f"Validation loss: {final_valid_loss_total:.4f} | Validation accuracy: {final_valid_accuracy:.4f}")
        print(f"      Test loss: {final_test_loss_total:.4f} |       Test accuracy: {final_test_accuracy:.4f}")

        save_dir_epoch_specific = os.path.join(save_dir,f"{args.task}_{args.name}")

        metrics = {
            'train_acc': final_train_accuracy,
            'valid_acc': final_valid_accuracy,
            'test_acc': final_test_accuracy,
            'train_loss': final_train_loss_total,
            'valid_loss': final_valid_loss_total,
            'test_loss': final_test_loss_total
        }

        if final_valid_accuracy > best_valid_acc:
            best_valid_acc = final_valid_accuracy
            patience_counter = 0
            print(f"New best validation accuracy:{best_valid_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy - Patience Counter:{patience_counter}/{patience_limit}")
        
        if patience_counter>=patience_limit:
            print(f"Early stopping triggered after epoch:{epoch+1}")
            break

if __name__ == "__main__":
   main()
   rpc.shutdown()