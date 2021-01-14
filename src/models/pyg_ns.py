"""
Defining GNN models (with node-sampling)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, NNConv
from src.models.utils import init_weights, get_act_fn


def define_ns_gnn_encoder(gnn_name):
    """
    return specified model class for GNNs with node-sampling.
    """
    if gnn_name == 'gat':
        return SamplingGAT
    elif gnn_name == 'sage':
        return SAGE
    elif gnn_name == 'mpnn':
        return SamplingMPNN
    else:
        raise NotImplementedError("node sampling only implemented for GAT, SAGE and MPNN models.")


def determine_fc_in_dim(config):
    """
    return dimensions of layers
    """
    flat_after = config['flat_after']
    add_lstm = config['add_last_ts']
    flat_dim = config['flat_nhid'] if config['flat_nhid'] is not None else config['num_flat_feats']
    if flat_after and add_lstm:
        in_dim = config['gnn_outdim'] + flat_dim + config['lstm_last_ts_dim']
    elif flat_after:
        in_dim = config['gnn_outdim'] + flat_dim
    else:
        in_dim = config['gnn_outdim'] + flat_dim
    return flat_after, add_lstm, in_dim, flat_dim


class NsGNN(nn.Module):
    """
    Model class for GNN with node-sampling scheme.
    """
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = define_ns_gnn_encoder(config['gnn_name'])(config)
        self.last_act = get_act_fn(config['final_act_fn'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward(self, x, flat, adjs, edge_weight):
        gnn_out = self.gnn_encoder.forward(x, flat, adjs, edge_weight)
        out = self.last_act(gnn_out)
        return out
    
    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight):
        out = self.gnn_encoder.inference(x_all, flat_all, subgraph_loader, device, edge_weight)
        out = self.last_act(out)
        return out


class SAGE(torch.nn.Module):
    """
    Model class for SAGE with node sampling scheme.
    """
    def __init__(self, config):
        super().__init__()

        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(config['gnn_indim'], config['sage_nhid']))
        self.convs.append(SAGEConv(config['sage_nhid'], config['gnn_outdim']))
        self.main_dropout = config['main_dropout']

        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
    
    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = flat[bsz_nids] if bsz_nids is not None else flat
            flat_bsz = self.flat_fc(flat_bsz)
            toc.append(flat_bsz)
        if self.add_lstm:
            if bsz_nids is not None:
                toc.append(last[bsz_nids])
            else:
                toc.append(last)
        return toc

    def forward(self, x, flat, adjs, edge_weight=None, last=None):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        

        toc = self.to_concat_vector(x, flat, last)
        if self.flat_after or self.add_lstm:
            x = torch.cat(toc, dim=1)
            x = F.dropout(x, p=self.main_dropout, training=self.training)
            x = self.out_layer(x)
        return x

    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight=None, last_all=None, get_emb=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                elif cat_in_last_layer: # last layer
                    bsz_nids = n_id[:batch_size]
                    toc = self.to_concat_vector(x, flat_all, last_all, bsz_nids)
                    x = torch.cat(toc, dim=1)
                    x = self.out_layer(x)
                    
                xs.append(x)
            x_all = torch.cat(xs, dim=0)

        return x_all


class SamplingGAT(torch.nn.Module):
    """
    Model class for GAT with node sampling scheme.
    """
    def __init__(self, config):
        super().__init__()

        self.featdrop = config['gat_featdrop']
        self.num_layers = 2
        in2 = config['gat_nhid']*config['gat_n_heads']
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(config['gnn_indim'], config['gat_nhid'], \
            heads=config['gat_n_heads'], dropout=config['gat_attndrop']))
        self.convs.append(GATConv(in2, config['gnn_outdim'], \
            heads=config['gat_n_out_heads'], concat=False, dropout=config['gat_attndrop']))
        self.main_dropout = config['main_dropout']
        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
    
    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = flat[bsz_nids] if bsz_nids is not None else flat
            flat_bsz = self.flat_fc(flat_bsz)
            toc.append(flat_bsz)
        if self.add_lstm:
            if bsz_nids is not None:
                toc.append(last[bsz_nids])
            else:
                toc.append(last)
        return toc

    def forward(self, x, flat, adjs, edge_weight=None, last=None):
        x = F.dropout(x, p=self.featdrop, training=self.training)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
        toc = self.to_concat_vector(x, flat, last)
        if self.flat_after or self.add_lstm:
            x = torch.cat(toc, dim=1)
            x = F.dropout(x, p=self.main_dropout, training=self.training)
            x = self.out_layer(x)
        return x

    def inference_whole(self, x_all, flat_all, device, edge_weight=None, edge_index=None, last_all=None, get_emb=False, get_attn=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        all_edge_attn = []
        x = x_all
        for i in range(self.num_layers):
            print('layer ',i )
            x, attn = self.convs[i].forward(x, edge_index, return_attention_weights=True)
            edge_attn = attn[1]
            all_edge_attn.append(edge_attn)
            if i != self.num_layers - 1:
                x = F.elu(x)
            elif cat_in_last_layer: # last layer
                toc = self.to_concat_vector(x, flat_all, last_all)
                x = torch.cat(toc, dim=1)
                x = self.out_layer(x)

        edge_index_w_self_loops = attn[0]

        return x, edge_index_w_self_loops, all_edge_attn

    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight=None, last_all=None, get_emb=False, get_attn=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        all_edge_attn = []
        for i in range(self.num_layers):
            edge_index_w_self_loops = []
            edge_attn = []
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if get_attn:
                    x, attn = self.convs[i].forward((x, x_target), edge_index, return_attention_weights=True)
                    edge_index_w_self_loops.append(attn[0])
                    edge_attn.append(attn[1])
                else:
                    x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.elu(x)
                elif cat_in_last_layer: # last layer
                    bsz_nids = n_id[:batch_size]
                    toc = self.to_concat_vector(x, flat_all, last_all, bsz_nids)
                    x = torch.cat(toc, dim=1)
                    x = self.out_layer(x)

                xs.append(x)
            x_all = torch.cat(xs, dim=0)
            if i == 1:
                edge_index_w_self_loops = torch.cat(edge_index_w_self_loops, dim=1) # [2, n. of edges]
            
            edge_attn = torch.cat(edge_attn, dim=0) # [no. of edges, n_heads of that layer]

            all_edge_attn.append(edge_attn)

        return x_all, edge_index_w_self_loops, all_edge_attn



class SamplingMPNN(torch.nn.Module):
    """
    Model class for MPNN with node sampling scheme.
    """
    def __init__(self, config):
        super(SamplingMPNN, self).__init__()
        dim = config['mpnn_nhid']
        self.lin0 = torch.nn.Linear(config['gnn_indim'], dim)
        nn_layers = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn_layers, aggr='mean')
        self.gru = nn.GRU(dim, dim)
        self.steps = config['mpnn_step_mp']
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, config['gnn_outdim'])
        self.main_dropout = config['main_dropout']
        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
        

    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = flat[bsz_nids] if bsz_nids is not None else flat
            flat_bsz = self.flat_fc(flat_bsz)
            toc.append(flat_bsz)
        if self.add_lstm:
            if bsz_nids is not None:
                toc.append(last[bsz_nids])
            else:
                toc.append(last)
        return toc

    def forward(self, x, flat, adjs, edge_weight, last=None):
        x = F.relu(self.lin0(x))

        edge_index = adjs[0]
        edge_ids = adjs[1]
        size = adjs[2]
        x_target = x[:size[1]]
        hid = x_target.unsqueeze(0)
        for step in range(self.steps):
            m = F.relu(self.conv((x, x_target), edge_index, edge_weight[edge_ids]))
            out, hid = self.gru(m.unsqueeze(0), hid)
            out = out.squeeze(0)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        # 
        toc = self.to_concat_vector(out, flat, last)
        if self.flat_after or self.add_lstm:
            out = torch.cat(toc, dim=1)
            out = F.dropout(out, p=self.main_dropout, training=self.training)
            out = self.out_layer(out)
        return out

    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight, last_all=None, get_emb=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        xs = []
        x_all = F.relu(self.lin0(x_all))
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, edge_ids, size = adj.to(device)
            x = x_all[n_id].to(device)
            x_target = x[:size[1]]
            hid = x_target.unsqueeze(0)
            for s in range(self.steps):
                m = F.relu(self.conv((x, x_target), edge_index, edge_weight[edge_ids]))
                out, hid = self.gru(m.unsqueeze(0), hid)
                out = out.squeeze(0)
            out = F.relu(self.lin1(out))
            out = self.lin2(out)
            # last layer
            if cat_in_last_layer:
                bsz_nids = n_id[:batch_size]
                toc = self.to_concat_vector(out, flat_all, last_all, bsz_nids)
                x = torch.cat(toc, dim=1)
                out = self.out_layer(x)
            xs.append(out)
        x_all = torch.cat(xs, dim=0)
        return x_all
    