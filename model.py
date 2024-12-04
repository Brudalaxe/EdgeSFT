import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertModel

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.int64)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.int64)
        
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

        self.embedding_dropout = nn.Dropout(p=0.5)
        self.classification_dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        emb_linear = self.embedding_dropout(emb_linear)

        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        
        y.last_hidden_state = self.classification_dropout(y.last_hidden_state)

        return y
    
    def get_rand_tok(self):
        c1,c2,c3,c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array([random.choice(range(c1)),random.choice(range(c2)),random.choice(range(c3)),random.choice(range(c4))])

if __name__ == "__main__":
    import pickle
    from transformers import BertConfig
    from data import prepare_data
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

    print("=== Testing Model Setup and Data Flow ===")

    class Args:
        def __init__(self):
            self.dataset = 'composer'
            self.task = 'composer'
            self.batch_size = 12
            self.num_workers = 5
            self.data_root = './data/dict/CP_data'
            self.max_seq_len = 512
            self.hs = 768

    args = Args()

    dict_path = './data/dict/CP.pkl'
    print("\nLoading Dictionary...")
    with open(dict_path, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nCreating model configuration...")
    config = BertConfig(
        max_position_embeddings=args.max_seq_len,
        position_embedding_type='relative_key_query',
        hidden_size=args.hs
    )

    print("\nInitialising model...")
    model = MidiBert(bertConfig=config, e2w=e2w, w2e=w2e)
    print(f"Model initialised with {sum(p.numel() for p in model.parameters())} parameters")

    checkpoint_path = './pretrain/pretrain_model.ckpt'
    print(f"\nLoading pretrained checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    print("\nLoading data...")
    data_loaders, data_info = prepare_data(args)
    train_loader, valid_loader, test_loader = data_loaders
    
    print("\nGetting a batch of data...")
    batch_data, batch_labels = next(iter(train_loader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = model.to(device)
    batch_data = batch_data.to(device)
    attention_mask = torch.ones((batch_data.shape[0], batch_data.shape[1])).to(device)

    print("\nTesting forward pass with real data...")
    try:
        with torch.no_grad():
            output = model(batch_data, attention_mask)
            
        print("\nModel Output Info:")
        print(f"Hidden states shape: {output.hidden_states[-1].shape}")
        print(f"Last layer output shape: {output.last_hidden_state.shape}")
        print("\nForward pass successful!")
        
        print("\nTesting layer-specific output:")
        for i, hidden_state in enumerate(output.hidden_states):
            print(f"Layer {i} output shape: {hidden_state.shape}")
            
    except Exception as e:
        print(f"Error in forward pass: {e}")

    print("\n=== Test Complete ===")