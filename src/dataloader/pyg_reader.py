"""
dataloaders to work with graphs
"""
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from src.dataloader.ts_reader import collect_ts_flat_labels, get_class_weights


def _sample_mask(idx, l):
    """Create sample mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def get_edge_index(us, vs, scores=None):
    """
    return edge data according to pytorch-geometric's specified formats.
    """
    both_us = np.concatenate([us, vs]) # both directions
    both_vs = np.concatenate([vs, us]) # both directions
    edge = np.stack([both_us, both_vs], 0)
    edge_index = torch.tensor(edge, dtype=torch.long)
    if scores is None:
        num_edges = edge_index.shape[1]
        scores = np.random.rand(num_edges, 1)
    else:
        scores = np.concatenate([scores, scores])[:, None]
    scores = torch.tensor(scores).float()
    return edge_index, scores


def get_rdm_edge_index(N, factor=2):
    """
    return random edge data
    """
    n_edge = N * factor
    us = np.random.choice(range(N), n_edge)
    vs = np.random.choice(range(N), n_edge)
    return get_edge_index(us, vs)


def define_node_masks(N, train_n, val_n):
    """
    define node masks according to train / val / test split
    """
    idx_train = range(train_n)
    idx_val = range(train_n, train_n + val_n)
    idx_test = range(train_n + val_n, N)
    train_mask = torch.BoolTensor(_sample_mask(idx_train, N))
    val_mask = torch.BoolTensor(_sample_mask(idx_val, N))
    test_mask = torch.BoolTensor(_sample_mask(idx_test, N))
    return train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


def read_txt(path, node=True):
    """
    read raw txt file into lists
    """
    u = open(path, "r")
    u_list = u.read()
    if node:
        return [int(n) for n in u_list.split('\n') if n != '']
    else:
        return [float(n) for n in u_list.split('\n') if n != '']

def read_graph_edge_list(graph_dir, version):
    """
    return edge lists, and edge similarity scores from specified graph.
    """
    version2filename = {'default': 'k_closest_{}_k=3_adjusted_ns.txt'}

    file_name = version2filename[version]
    u_path = Path(graph_dir) / file_name.format('u')
    v_path = Path(graph_dir) / file_name.format('v')
    scores_path = Path(graph_dir) / file_name.format('scores')
    u_list = read_txt(u_path)
    v_list = read_txt(v_path)
    if os.path.exists(scores_path):
        scores = read_txt(scores_path, node=False)
    else:
        scores = None
    return u_list, v_list, scores


class GraphDataset(Dataset):
    """
    Dataset class for graph data
    """
    def __init__(self, config, us=None, vs=None):
        super().__init__()
        
        data_dir = config['data_dir']
        task = config['task']

        # Get node features
        seq, flat, labels, info, N, train_n, val_n = collect_ts_flat_labels(data_dir, \
                                config['ts_mask'], task, config['add_diag'], debug=config['debug'])
        self.info = info
        
        # Get the edges
        if config['model'] == 'lstm':
            edge_index, edge_attr = get_rdm_edge_index(N, 1)
        else:
            if config['random_g']:
                edge_index, edge_attr = get_rdm_edge_index(N)
            else:
                us, vs, edge_attr = read_graph_edge_list(config['graph_dir'], config['g_version'])
                edge_index, edge_attr = get_edge_index(us, vs, edge_attr)
        
        # Record feature dimensions
        self.ts_dim = seq.shape[1] if config['read_lstm_emb'] else seq.shape[2]
        self.flat_dim = flat.shape[1]
        x = seq
        if config['flatten']:
            x = np.reshape(x, (len(x), -1))
            if config['flat_first'] and config['add_flat']:
                x = np.concatenate([x, flat], 1)
            self.x_dim = x.shape[1]
        else:
            self.x_dim = self.ts_dim
        
        if config['verbose']:
            print(f'Dimensions of ts: {self.ts_dim}, flat features: {self.flat_dim}, x: {self.x_dim}')
        
        # define the graph and its features
        x = torch.from_numpy(x).float()
        flat = torch.from_numpy(flat).float()
        y = torch.from_numpy(labels)
        y = y.long() if task == 'ihm' else y.float()
        data = Data(x=x, edge_index=edge_index, y=y, flat=flat, edge_attr=edge_attr)

        # define masks
        data.train_mask, data.val_mask, data.test_mask, self.idx_train, self.idx_val, self.idx_test = define_node_masks(N, train_n, val_n)
        self.data = data

        # define class weights
        self.class_weights = get_class_weights(labels[:train_n]) if task == 'ihm' else False

    
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return self.data.x, self.data.flat, self.data.edge_index, self.data.edge_attr, self.data.y
