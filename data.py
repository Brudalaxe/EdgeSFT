import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FinetuneDataset(Dataset):
    """Dataset class for MIDI-BERT fine-tuning tasks"""
    def __init__(self, X, y):
        self.data = X 
        self.label = y

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])

def load_data(dataset, task, data_root='./data/dict/CP_data'):
    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia']:
        raise ValueError(f'Dataset {dataset} not supported')
        
    X_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True)
    
    if dataset == 'pop909':
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=12, num_workers=5):
    train_dataset = FinetuneDataset(X=X_train, y=y_train)
    valid_dataset = FinetuneDataset(X=X_val, y=y_val)
    test_dataset = FinetuneDataset(X=X_test, y=y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    
    return train_loader, valid_loader, test_loader

def get_data_info(dataset, task):
    if task in ['melody', 'velocity']:
        seq_class = False
        num_classes = 4 if task == 'melody' else 7
    elif task == 'composer':
        seq_class = True
        num_classes = 8
    elif task == 'emotion':
        seq_class = True
        num_classes = 4
    else:
        raise ValueError(f"Task {task} not supported")
        
    return seq_class, num_classes

def prepare_data(args):
    seq_class, num_classes = get_data_info(args.dataset, args.task)
    
    X_data, y_data = load_data(args.dataset, args.task, args.data_root)
    
    data_loaders = create_data_loaders(
        *X_data, *y_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    data_info = {
        'seq_class': seq_class,
        'num_classes': num_classes,
        'train_size': len(X_data[0]),
        'val_size': len(X_data[1]),
        'test_size': len(X_data[2])
    }
    
    return data_loaders, data_info

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.dataset = 'composer'
            self.task = 'composer'
            self.batch_size = 12
            self.num_workers = 5
            self.data_root = './data/dict/CP_data'

    args = Args()
    
    print("=== Testing Data Loading ===")
    
    print("\nTesting get_data_info:")
    seq_class, num_classes = get_data_info(args.dataset, args.task)
    print(f"Sequence classification: {seq_class}")
    print(f"Number of classes: {num_classes}")
    
    print("\nTesting load_data:")
    try:
        X_data, y_data = load_data(args.dataset, args.task, args.data_root)
        print("X shapes:", [x.shape for x in X_data])
        print("y shapes:", [y.shape for y in y_data])
    except Exception as e:
        print(f"Error loading data: {e}")
    
    print("\nTesting full data preparation:")
    try:
        data_loaders, data_info = prepare_data(args)
        train_loader, valid_loader, test_loader = data_loaders
        
        print("\nDataLoader Information:")
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Valid loader batches: {len(valid_loader)}")
        print(f"Test loader batches: {len(test_loader)}")
        
        print("\nData Info:")
        for key, value in data_info.items():
            print(f"{key}: {value}")
        
        print("\nTesting batch loading:")
        batch_data, batch_labels = next(iter(train_loader))
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")