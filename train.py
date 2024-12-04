import argparse
import os
import time
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertConfig
from tqdm import tqdm
import wandb

from model import MidiBert
from data import prepare_data

def get_args():
    parser = argparse.ArgumentParser(description='Single GPU training for MIDI-BERT')
    
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'], 
                        required=True, help='Fine-tuning task')
    
    parser.add_argument('--dict_file', type=str, 
                        default='./data/dict/CP.pkl')
    parser.add_argument('--ckpt', 
                        default='./pretrain/pretrain_model.ckpt')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--data_root', type=str, 
                        default='./data/dict/CP_data')
    
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=2021)
    
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.task in ['melody', 'velocity']:
        args.dataset = 'pop909'
    elif args.task == 'composer':
        args.dataset = 'composer'
    elif args.task == 'emotion':
        args.dataset = 'emopia'
        
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc='  Training')
    for step, (batch_data, batch_labels) in enumerate(progress_bar):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        attention_mask = torch.ones((batch_data.shape[0], batch_data.shape[1])).to(device)
        
        outputs = model(batch_data, attention_mask)
        logits = outputs.last_hidden_state.mean(dim=1)
        
        loss = criterion(logits, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        acc = (predictions == batch_labels).float().mean()
        
        total_loss += loss.item() * batch_data.size(0)
        total_acc += acc.item() * batch_data.size(0)
        total_samples += batch_data.size(0)

        wandb.log({
            "train/loss": loss.item(),
            "train/accuracy": acc.item(),
            "train/step": step + epoch * len(train_loader)
        })
        
        progress_bar.set_postfix({'Loss': loss.item(), 'acc': acc.item()})
    
    return total_loss / total_samples, total_acc / total_samples

def evaluate(model, data_loader, criterion, device, epoch, prefix):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    desc = 'Validating' if prefix == 'val' else '   Testing'
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc)
        for batch_data, batch_labels in progress_bar:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            attention_mask = torch.ones((batch_data.shape[0], batch_data.shape[1])).to(device)
            
            outputs = model(batch_data, attention_mask)
            logits = outputs.last_hidden_state.mean(dim=1)
            
            loss = criterion(logits, batch_labels)
            predictions = torch.argmax(logits, dim=1)
            acc = (predictions == batch_labels).float().mean()
            
            total_loss += loss.item() * batch_data.size(0)
            total_acc += acc.item() * batch_data.size(0)
            total_samples += batch_data.size(0)

            progress_bar.set_postfix({'loss': loss.item(), 'acc': acc.item()})

    epoch_loss = total_loss / total_samples
    epoch_acc = total_acc / total_samples
    
    wandb.log({
        f"{prefix}/loss": epoch_loss,
        f"{prefix}/accuracy": epoch_acc,
        "epoch": epoch
    })
    
    return total_loss / total_samples, total_acc / total_samples

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Safe checkpoint saving function"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        temp_file = filepath + '.tmp'
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                **metrics
            }
        except Exception as opt_err:
            print(f"Warning: Could not save optimizer state: {str(opt_err)}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                **metrics
            }
        
        torch.save(checkpoint, temp_file, _use_new_zipfile_serialization=True)
        
        os.replace(temp_file, filepath)
        print(f"Checkpoint saved successfully to {filepath}")
        
    except Exception as e:
        print(f"Warning: Could not save checkpoint to {filepath}")
        print(f"Error: {str(e)}")
        
        try:
            torch.save(
                {'model_state_dict': model.state_dict()},
                filepath + '.model_only',
                _use_new_zipfile_serialization=True
            )
            print(f"Saved model-only checkpoint to {filepath}.model_only")
        except Exception as fallback_err:
            print(f"Could not save model-only checkpoint: {str(fallback_err)}")

def main():
    args = get_args()
    set_seed(args.seed)

    wandb.init(
        project="MidiBert_SFT",
        group="full_finetuning",
        name=f"run_Full_Finetuning_{args.name}",
        config={
            "model_type": "Full Fine-tuning",
            "learning_rate": 2e-5,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model": "BERT-Base",
            "task": args.task,
            "architecture": args.name
        }
    )
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(f'Using device: {device}')

    save_dir = os.path.join(args.output_dir, f"{args.task}_{args.name}")
    os.makedirs(save_dir, exist_ok=True)

    print("\nLoading Dictionary...")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    
    data_loaders, data_info = prepare_data(args)
    train_loader, valid_loader, test_loader = data_loaders
    
    config = BertConfig(
        max_position_embeddings=args.max_seq_len,
        position_embedding_type='relative_key_query',
        hidden_size=args.hidden_size
    )
    
    model = MidiBert(bertConfig=config, e2w=e2w, w2e=w2e)

    wandb.define_metric("train_accuracy", summary="max")
    wandb.define_metric("val_accuracy", summary="max")
    wandb.define_metric("test_accuracy", summary="max")
    
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained weights from {args.ckpt}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.3, betas=(0.9, 0.999), eps=1e-8)
    
    best_valid_acc = 0
    patience = 3
    patience_counter = 0
    
    training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device, epoch, prefix="val")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch, prefix="test")
        
        print(f"Epoch {epoch+1} completed.")

        print(f'Epoch {epoch+1} Summary')
        print(f"  Training loss: {train_loss:.4f} |   Training accuracy: {(train_acc * 100):.2f}%")
        print(f'Validation loss: {valid_loss:.4f} | Validation accuracy: {(valid_acc * 100):.2f}%')
        print(f'      Test loss: {test_loss:.4f} |       Test accuracy: {(test_acc * 100):.2f}%')
        
        metrics = {
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_loss': test_loss
        }
        
        try:
            save_checkpoint(
                model, 
                optimizer, 
                epoch,
                metrics,
                os.path.join(save_dir, 'last_model.pt')
            )
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    os.path.join(save_dir, 'best_model.pt')
                )
            else:
                patience_counter += 1
                
        except Exception as e:
            print(f"Warning: Failed to save checkpoints: {str(e)}")
            print("Continuing training without saving...")
        
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Calculate and print total training time
    training_end_time = time.perf_counter()
    total_training_time = training_end_time - training_start_time

    # Convert to hours, minutes, seconds
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = total_training_time % 60

    print(f"\nTotal training time: {hours}h {minutes}m {seconds:.2f}s")

if __name__ == '__main__':
    main()