"""
Defining Dyanmics LSTM-GNN models
"""
import torch
import torch.nn as nn
from src.models.lstm import define_lstm_encoder
from src.models.pyg_whole import define_gnn_encoder
from src.models.pyg_ns import determine_fc_in_dim
from src.models.utils import init_weights, get_act_fn


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x, as_tuple=False)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


class DynamicLstmGnn(nn.Module):
    """model class for Dynamic LSTM-GNN"""
    def __init__(self, config):
        super().__init__()
        self.lstm_encoder = define_lstm_encoder()(config)
        self.gnn_encoder = define_gnn_encoder(config['gnn_name'])(config)
        self.k = config['dg_k']
        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
        self.last_act = get_act_fn(config['final_act_fn'])
        self.drop = nn.Dropout(config['main_dropout'])
        self.lstm_out = nn.Linear(config['lstm_last_ts_dim'], config['out_dim'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = self.flat_fc(flat)
            toc.append(flat_bsz)
        if self.add_lstm:
            toc.append(last)
        return toc

    def knn_to_graph(self, x):
        inner = -2*torch.matmul(x, x.transpose(1, 0))
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(1, 0)
        k_closest = pairwise_distance.topk(k=self.k, dim=-1)   # (batch_size, num_points, k, 2)
        # to graph list
        us, vs, vals = self.get_u_v(k_closest[1], k_closest[0])
        edge_index = torch.stack([us, vs], axis=0).float()
        edge_index = edge_index.long()
        vals = vals[:, None].float()
        return edge_index, vals

    def get_u_v(self, idx, edge_weights):
        B = len(idx)
        idx_prime = torch.Tensor(range(B)).unsqueeze(1).repeat_interleave(self.k, dim=1).long()
        idx_prime = idx_prime.cuda(idx.get_device()) if idx.is_cuda else idx_prime
        edges = torch.sparse.FloatTensor(
                indices=torch.cat((idx_prime.flatten().unsqueeze(0), idx.flatten().unsqueeze(0)), axis=0),
                values=edge_weights.flatten())
        edges = edges.to_dense()
        edges_int = (edges > 0).short()
        reflected_int = edges_int + edges_int.transpose(1, 0)
        reflected_edges = edges + edges.transpose(1, 0)
        reflected_int[reflected_int == 0] = 1  # make sure I'm not dividing by zero
        edges = reflected_edges / reflected_int  # symmetric matrix
        edges.fill_diagonal_(0)  # remove self connections
        sparse_edges = to_sparse(edges).coalesce()
        us, vs = sparse_edges.indices()
        vals = sparse_edges.values()
        return us, vs, vals

    def forward(self, x, flat):
        # lstm
        seq = x.permute(1, 0, 2)
        out, _ = self.lstm_encoder.forward(seq)
        last = out[:, -1, :] if len(out.shape)==3 else out
        out = out.view(out.size(0), -1) # bsz, lstm_outdim
        # knn
        edge_index, edge_weights = self.knn_to_graph(out)
        # gnn
        out = self.gnn_encoder.forward(out, edge_index, edge_weights)

        toc = self.to_concat_vector(out, flat, last)
        if self.flat_after or self.add_lstm:
            out = torch.cat(toc, dim=1)
            out = self.out_layer(self.drop(out))
        out = self.last_act(out)

        # predict from lstm too
        lstm_y = self.last_act(self.lstm_out(last))

        return out, lstm_y