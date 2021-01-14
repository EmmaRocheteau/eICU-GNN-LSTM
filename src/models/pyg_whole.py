"""
Defining GNN models (without node-sampling)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv
from src.models.utils import init_weights, get_act_fn


def define_gnn_encoder(gnn_name):
    """
    return specified model class for GNNs (without node-sampling).
    """
    if gnn_name == 'gcn':
        return GCN
    elif gnn_name == 'gat':
        return GAT
    elif gnn_name == 'mpnn':
        return MPNN


class WholeGNN(nn.Module):
    """
    Model class for GNNs without node-sampling.
    """
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = define_gnn_encoder(config['gnn_name'])(config)
        self.last_act = get_act_fn(config['final_act_fn'])
        # where to put the flat features
        # self.flat_before = config['add_flat'] and config['flat_first'] (done in GraphDataset)
        self.flat_after = config['flat_after']
        if self.flat_after:
            flat_dim = config['flat_nhid'] if config['flat_nhid'] is not None else config['num_flat_feats']
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
            fc_in_dim = config['gnn_outdim'] + flat_dim
            self.out_layer = nn.Linear(fc_in_dim, config['out_dim'])
        self.drop = nn.Dropout(config['main_dropout'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())
        
    def forward(self, x, flat, edge_index, edge_weight=None):
        out = self.gnn_encoder.forward(x, edge_index, edge_weight)
        
        if self.flat_after:
            flat = self.flat_fc(flat)
            out = torch.cat([out, flat], dim=1)
            out = self.out_layer(self.drop(out))
        out = self.last_act(out)
        return out        


class GCN(torch.nn.Module):
    """
    Model class for GCN.
    """
    def __init__(self, config):
        super().__init__()
        self.conv1 = GCNConv(config['gnn_indim'], config['gcn_nhid'], cached=False)
        self.conv2 = GCNConv(config['gcn_nhid'], config['gnn_outdim'], cached=False)
        self.use_edge_weight = config['edge_weight']
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index, edge_weight=None):
        edge_weight = edge_weight.view(-1) if self.use_edge_weight else None
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # return F.log_softmax(x, dim=1)
        return x


class GAT(torch.nn.Module):
    """
    Model class for GAT.
    """
    def __init__(self, config):
        super().__init__()
        in2 = config['gat_nhid']*config['gat_n_heads']
        self.conv1 = GATConv(config['gnn_indim'], config['gat_nhid'], \
            heads=config['gat_n_heads'], dropout=config['gat_attndrop'])

        self.conv2 = GATConv(in2, config['gnn_outdim'], \
            heads=config['gat_n_out_heads'], concat=False, dropout=config['gat_attndrop'])

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class MPNN(torch.nn.Module):
    """
    Model class for MPNN.
    """
    def __init__(self, config):
        super(MPNN, self).__init__()
        dim = config['mpnn_nhid']
        self.lin0 = torch.nn.Linear(config['gnn_indim'], dim)
        nn_layers = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn_layers, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.steps = config['mpnn_step_mp']
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, config['gnn_outdim'])

    def forward(self, x, edge_index, edge_weight):
        out = F.relu(self.lin0(x))
        hid = out.unsqueeze(0)
        for step in range(self.steps):
            m = F.relu(self.conv(out, edge_index, edge_weight))
            out, hid = self.gru(m.unsqueeze(0), hid)
            out = out.squeeze(0)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out