"""
Defining LSTM-GNN models
"""
from tqdm import tqdm
import torch
import torch.nn as nn
from src.models.lstm import define_lstm_encoder
from src.models.pyg_ns import define_ns_gnn_encoder
from src.models.utils import init_weights, get_act_fn


class NsLstmGNN(torch.nn.Module):
    """
    model class for LSTM-GNN with node-sampling scheme.
    """
    def __init__(self, config):
        super().__init__()
        self.lstm_pooling = config['lstm_pooling']
        self.lstm_encoder = define_lstm_encoder()(config)
        self.gnn_name = config['gnn_name']
        self.gnn_encoder = define_ns_gnn_encoder(config['gnn_name'])(config)
        self.last_act = get_act_fn(config['final_act_fn'])
        self.lstm_out = nn.Linear(config['lstm_last_ts_dim'], config['out_dim'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward(self, x, flat, adjs, batch_size, edge_weight):
        seq = x.permute(1, 0, 2)
        out, _ = self.lstm_encoder.forward(seq)
        last = out[:, -1, :] if len(out.shape)==3 else out
        last = last[:batch_size]
        out = out.view(out.size(0), -1) # all_nodes, lstm_outdim
        x = out
        x = self.gnn_encoder(x, flat, adjs, edge_weight, last)
        y = self.last_act(x)        
        lstm_y = self.last_act(self.lstm_out(last))
        return y, lstm_y

    def infer_lstm_by_batch(self, ts_loader, device):
        lstm_outs = []
        lasts = []
        lstm_ys = []
        for inputs, labels, ids in ts_loader:
            seq, flat = inputs
            seq = seq.to(device)
            seq = seq.permute(1, 0, 2)
            out, _ = self.lstm_encoder.forward(seq)
            last = out[:, -1, :] if len(out.shape)==3 else out
            out = out.view(out.size(0), -1)
            lstm_y = self.last_act(self.lstm_out(last))
            lstm_outs.append(out)
            lasts.append(last)
            lstm_ys.append(lstm_y)
        lstm_outs = torch.cat(lstm_outs, dim=0) # [entire_g, dim]
        lasts = torch.cat(lasts, dim=0) # [entire_g, dim]
        lstm_ys = torch.cat(lstm_ys, dim=0)
        print('Got all LSTM output.')
        return lstm_outs, lasts, lstm_ys

    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False):
        # first collect lstm outputs by minibatching:
        lstm_outs, last_all, lstm_ys = self.infer_lstm_by_batch(ts_loader, device)

        # then pass lstm outputs to gnn
        x_all = lstm_outs
        out = self.gnn_encoder.inference(x_all, flat_all, subgraph_loader, device, edge_weight, last_all, get_emb=get_emb)

        out = self.last_act(out)

        return out, lstm_ys

    def inference_w_attn(self, x_all, flat_all, edge_weight, edge_index, ts_loader, subgraph_loader, device):
        lstm_outs, last_all, lstm_ys = self.infer_lstm_by_batch(ts_loader, device)
        x_all = lstm_outs
        ret = self.gnn_encoder.inference_whole(x_all, flat_all, device, edge_weight, edge_index, last_all, get_attn=True)
        x_all, edge_index_w_self_loops, all_edge_attn = ret

        out = self.last_act(x_all)

        return out, lstm_ys, edge_index_w_self_loops, all_edge_attn

