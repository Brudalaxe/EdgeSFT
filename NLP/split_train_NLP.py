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
from split_model_NLP import BertFrontOri, BertFrontFFN, BertFrontDecomposition, BertFrontOriQuant, BertFrontFFNQuant, BertFrontFFNQuantRes, DistMidiBertFFNQuantRes, DistBert
from data import TextClassificationDataset
import argparse
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Training for split BERT')
    parser.add_argument('--pretrain_dir', type=str, default='/home/brad/MidiBERT/pretrain/bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
    parser.add_argument('--train_path', type=str, default='/home/brad/MidiBERT/data/SST-2/train.tsv')
    parser.add_argument('--dev_path', type=str, default='/home/brad/MidiBERT/data/SST-2/dev.tsv')
    parser.add_argument('--max_length', type=int, default=66)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--split_layer', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    return parser.parse_args()

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
        
def load_model_and_classifier_v2(model_choice, pretrain_dir, split_layer, rank, label_nums, devices):
    model_mapping = {
        1: {"name": "Split Learning (Original)", 
            "loader": BertFrontOri},
        2: {"name": "Split Fine-tuning", 
            "loader": BertFrontFFN},
        3: {"name": "Split Fine-tuning with Decomposition", 
            "loader": BertFrontDecomposition},
        4: {"name": "Split Fine-tuning with Quantisation", 
            "loader": BertFrontOriQuant},
        5: {"name": "Split Fine-tuning (FFN) with Quantisation", 
            "loader": BertFrontFFNQuant},
        6: {"name": "Split Fine-tuning (FFN) with Quantisation and Residual", 
            "loader": BertFrontFFNQuantRes}
    }

    print(f"Using {model_mapping[model_choice]['name']} model")
    
    front_model = model_mapping[model_choice]["loader"](
        pretrain_dir=pretrain_dir,
        split_num=split_layer,
        rank=rank,
        device=devices["cloud"]
    )

    classifier = DistBert(
        cloud_pretrain_dir=pretrain_dir,
        edge_pretrain_dir=pretrain_dir,
        split_num=split_layer,
        rank=rank,
        label_nums=label_nums,
        devices=devices,
        model_type=model_choice
    )

    return classifier, front_model

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

    devices = {
        "edge": "cpu",
        "cloud": "cuda:0"
    }

    print("\nLoading data...")
    train_data = TextClassificationDataset(args.pretrain_dir)
    train_data.build_label_index()
    train_data.read_file(args.train_path, max_length=args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, 
                            collate_fn=train_data.collate, shuffle=True)

    dev_data = TextClassificationDataset(args.pretrain_dir)
    dev_data.build_label_index()
    dev_data.read_file(args.dev_path, max_length=args.max_length)
    dev_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, 
                          collate_fn=dev_data.collate)

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
    
    print("\nLoading split models...")
    classifier, front_model = load_model_and_classifier_v2(
        model_choice,
        args.pretrain_dir,
        args.split_layer,
        args.rank,
        train_data.label_nums,
        devices
    )

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
    epochs = args.epoch

    ### Training phase

    for epoch in range(epochs):
        classifier.train()
        
        total_train_loss = 0.0
        correct_train_predictions_total = 0
        total_train_samples = 0

        with tqdm(train_loader, desc=f"  Training") as train_tqdm:
            for i, batch in enumerate(train_tqdm):
                batch_input_ids = batch["batch_input_ids"]
                batch_token_type_ids = batch["batch_token_type_ids"]
                batch_attention_mask = batch["batch_attention_mask"]
                y_true = batch["batch_labels"].to(devices["edge"])

                with dist_autograd.context() as context_id:
                    y_pred = classifier(
                        input_ids=batch_input_ids,
                        token_type_ids=batch_token_type_ids,
                        attention_mask=batch_attention_mask
                    ).to(devices["edge"])
                    
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
        
        with tqdm(dev_loader, desc=f"Validating") as valid_tqdm:
            for batch in valid_tqdm:
                batch_input_ids = batch["batch_input_ids"]
                batch_token_type_ids = batch["batch_token_type_ids"]
                batch_attention_mask = batch["batch_attention_mask"]
                y_true = batch["batch_labels"].to(devices["edge"])

                with torch.no_grad():
                    y_pred = classifier(
                        input_ids=batch_input_ids,
                        token_type_ids=batch_token_type_ids,
                        attention_mask=batch_attention_mask
                    ).to(devices["edge"])
                    
                    loss_val = criterion(y_pred, y_true).item()
                    valid_loss_total += loss_val * len(batch_input_ids)
                    
                    predictions_val = torch.argmax(y_pred, dim=-1)
                    correct = torch.eq(predictions_val, y_true).sum().item()
                    correct_val_predictions_total += correct

                valid_tqdm.set_postfix({"Loss": loss_val})

        final_valid_loss_total = valid_loss_total / len(dev_loader.dataset)
        final_valid_accuracy = (correct_val_predictions_total / len(dev_loader.dataset))

        print(f"Epoch {epoch+1} completed.")
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training loss: {final_train_loss_total:.4f} |   Training accuracy: {final_train_accuracy:.4f}")
        print(f"Validation loss: {final_valid_loss_total:.4f} | Validation accuracy: {final_valid_accuracy:.4f}")

        if final_valid_accuracy > best_valid_acc:
            best_valid_acc = final_valid_accuracy
            patience_counter = 0
            print(f"New best validation accuracy: {best_valid_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy - Patience Counter: {patience_counter}/{patience_limit}")
        
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after epoch: {epoch+1}")
            break

if __name__ == "__main__":
   main()
   rpc.shutdown()