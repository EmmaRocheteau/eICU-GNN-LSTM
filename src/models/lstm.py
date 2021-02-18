"""
Defining LSTM models
"""
import torch
import torch.nn as nn
from src.models.utils import init_weights, get_act_fn


def define_lstm_encoder():
    return DynamicLSTM


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm_encoder = DynamicLSTM(config)
        self.flat_after = config['flat_after']
        fc_in_dim = config['lstm_outdim']
        if self.flat_after:
            flat_dim = config['flat_nhid'] if config['flat_nhid'] is not None else config['num_flat_feats']
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
            fc_in_dim += flat_dim
        self.out_layer = nn.Linear(fc_in_dim, config['out_dim'])
        self.drop = nn.Dropout(config['main_dropout'])
        self.last_act = get_act_fn(config['final_act_fn'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward_to_lstm(self, seq, flat):
        seq = seq.permute(1, 0, 2)
        out, _ = self.lstm_encoder.forward(seq)
        out = out.view(out.size(0), -1) # bsz, output_dim
        return out

    def forward(self, seq, flat):
        seq = seq.permute(1, 0, 2) # seq_len, bsz, in_dim
        out, _ = self.lstm_encoder.forward(seq)
        out = out.view(out.size(0), -1) # bsz, output_dim
        if self.flat_after:
            flat = self.flat_fc(flat) # bsz, flat_nhid
            out = torch.cat([out, flat], dim=1) # bsz, lstm_nhid + flat_nhid
        out = self.out_layer(self.drop(out))
        out = self.last_act(out)
        return out
    

class DynamicLSTM(nn.Module):
    """
    model class for LSTM
    """
    def __init__(self, config):

        super().__init__()

        self.n_layers = config['lstm_layers']
        self.dropout_rate = config['lstm_dropout']
        self.pooling = config['lstm_pooling']
        self.is_bidirectional = config['bilstm']

        if self.is_bidirectional:
            self.num_units = config['lstm_nhid'] // 2
            self.num_dir = 2
        else:
            self.num_units = config['lstm_nhid']
            self.num_dir = 1

        self.drop = nn.Dropout(self.dropout_rate)
        # lstm
        self.lstm = nn.LSTM(config['lstm_indim'], self.num_units, self.n_layers, 
                            dropout=self.dropout_rate, bidirectional=self.is_bidirectional)
        self.hidden = self.init_hidden(config['batch_size'])

    def init_hidden(self, batch_size):
        """
        init hidden state for LSTM 
        note to self:
        - this function seems unnecessary since nn.LSTM defaults to initialize with zeroes anyway.
        - weight.new creates a Varialbe with the same data type as weight (i.e. nth actually to do with weight)
        - init_hidden doesn't initialize the weights, rather, it creates new initial states for new sequences (t=0)
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * self.num_dir, batch_size, self.num_units).zero_(), 
                  weight.new(self.n_layers * self.num_dir, batch_size, self.num_units).zero_())
        return hidden

    def forward(self, lstm_input):
        """
        one forward step
        """
        # Assume: lstm_input of shape [seq_len, batch_size, feat_dim]
        self.hidden = self.init_hidden(lstm_input.shape[1])
        lstm_out, _ = self.lstm(lstm_input, self.hidden)   # (seq_len, batch_size, nhid)

        lstm_out = torch.transpose(lstm_out, 0, 1).contiguous()
        # lstm_out has # (batch_size, seq_len, nhid)
        # except 'all' pooling, all lstm_out becomes (batch_size, nhid). 
        if self.pooling == 'mean':
            lstm_out = torch.mean(lstm_out, 1).squeeze()
        elif self.pooling == 'max':
            lstm_out = torch.max(lstm_out, 1)[0].squeeze()
        elif self.pooling == 'last':
            if self.is_bidirectional:
                # only to be used when bidirectional.
                # concat last step of forward dir + first step of backward dir.
                lstm_out = torch.cat((lstm_out[:, -1, :self.num_units], lstm_out[:, 0, self.num_units:]), 1)
            else:
                lstm_out = lstm_out[:, -1, :]
        elif self.pooling == 'all':
            pass
        else:
            raise NotImplementedError('only pooling mean / all for now.')
            
        attention = None

        return lstm_out, attention
