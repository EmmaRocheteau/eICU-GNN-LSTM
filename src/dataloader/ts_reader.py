"""
Dataloaders for lstm_only model
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataloader.convert import read_mm


def slice_data(data, info, split):
    """Slice data according to the instances belonging to each split."""
    if split is None:
        return data
    elif split == 'train':
        return data[:info['train_len']]
    elif split == 'val':
        train_n = info['train_len']
        val_n = train_n + info['val_len']
        return data[train_n: val_n]
    elif split == 'test':
        val_n = info['train_len'] + info['val_len']
        test_n = val_n + info['test_len']
        return data[val_n: test_n]


def no_mask_cols(ts_info, seq):
    """do not apply temporal masks"""
    neg_mask_cols = [i for i, e in enumerate(ts_info['columns']) if 'mask' not in e]
    return seq[:, :, neg_mask_cols]


def collect_ts_flat_labels(data_dir, ts_mask, task, add_diag, split=None, 
                                debug=0, split_flat_and_diag=False):
    """
    read temporal, flat data and task labels
    """
    flat_data, flat_info = read_mm(data_dir, 'flat')
    flat = slice_data(flat_data, flat_info, split)

    ts_data, ts_info = read_mm(data_dir, 'ts')
    seq = slice_data(ts_data, ts_info, split)
    if not ts_mask:
        seq = no_mask_cols(ts_info, seq)

    if add_diag:
        diag_data, diag_info = read_mm(data_dir, 'diagnoses')
        diag = slice_data(diag_data, flat_info, split)
        if split_flat_and_diag:
            flat = (flat, diag)
        else:
            flat = np.concatenate([flat, diag], 1)

    label_data, labels_info = read_mm(data_dir, 'labels')
    labels = slice_data(label_data, flat_info, split)
    idx2col = {'ihm': 1, 'los': 3, 'multi': [1, 3]}
    label_idx = idx2col[task]
    labels = labels[:, label_idx]

    if debug:
        N = 1000
        train_n = int(N*0.5)
        val_n = int(N*0.25)
    else:
        N = flat_info['total']
        train_n = flat_info['train_len']
        val_n = flat_info['val_len']
    
    seq = seq[:N]
    flat = flat[:N]
    labels = labels[:N]
    return seq, flat, labels, flat_info, N, train_n, val_n


def get_class_weights(train_labels):
    """
    return class weights to handle class imbalance problems
    """
    occurences = np.unique(train_labels, return_counts=True)[1]
    class_weights = occurences.sum() / occurences
    class_weights = torch.Tensor(class_weights).float()
    return class_weights


class LstmDataset(Dataset):
    """
    Dataset class for temporal data.
    """
    def __init__(self, config, split=None):
        super().__init__()
        task = config['task']

        self.seq, self.flat, self.labels, self.ts_info, self.N, train_n, val_n = collect_ts_flat_labels(config['data_dir'], config['ts_mask'], \
                                                                                    task, config['add_diag'], split, debug=0)
        
        self.ts_dim = self.seq.shape[2]
        self.flat_dim = self.flat.shape[1]

        if split == 'train':
            self.split_n = train_n
        elif split == 'val':
            self.split_n = val_n
        elif split == 'test':
            self.split_n = self.N - train_n - val_n

        self.idx_val = range(train_n, train_n + val_n)
        self.idx_test = range(train_n + val_n, self.N)
        
        self.split_n = self.N if split is None else self.ts_info[f'{split}_len']
        all_nodes = np.arange(self.N)
        self.ids = slice_data(all_nodes, self.ts_info, split) # (N_split.)

        self.class_weights = get_class_weights(self.labels[:train_n]) if task == 'ihm' else False

    def __len__(self):
        return self.split_n
    
    def __getitem__(self, index):
        return self.seq[index], self.flat[index], self.labels[index], self.ids[index]


def collate_fn(x_list, task):
    """
    collect samples in each batch
    """
    seq = torch.Tensor(np.stack([sample[0] for sample in x_list])).float() # [bsz, seq_len, ts_dim]
    flat = torch.Tensor(np.stack([sample[1] for sample in x_list])).float() # [bsz, flat_dim]
    inputs = (seq, flat)
    if task == 'los':
        labels = torch.Tensor(np.stack([sample[2] for sample in x_list])).float() # [bsz,]
    else:
        labels = torch.Tensor(np.stack([sample[2] for sample in x_list])).long() # [bsz,]
    ids = torch.Tensor(np.stack([sample[3] for sample in x_list])).long()
    return inputs, labels, ids
