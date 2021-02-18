import argparse
import torch
import sys
from pathlib import Path
from src.hyperparameters.best_parameters import lstmgnn, dynamic, ns_gnn_default
from src.utils import load_json

def add_best_params(config):
    """
    read best set of hyperparams
    """
    if config['model'] == 'gnn':
        best = ns_gnn_default
    elif config['dynamic_g']:
        best = dynamic
    elif config['model'] == 'lstmgnn':
        best = lstmgnn
    best = best[config['task']][config['gnn_name']]
    for key, value in best.items():
        config[key] = value
    print('*** using best values for these params', [p for p in best])


def add_tune_params(parser):
    """
    define hyperparam-tuning params
    """
    parser.add_argument('--num_samples', type=int, default=15)
    parser.add_argument('--gpus_per_trial', type=int, default=1)
    parser.add_argument('--cpus_per_trial', type=int, default=7)
    parser.add_argument('--grace_period', type=int, default=3)
    parser.add_argument('--fix_g_params', action='store_true')
    parser.add_argument('--fix_l_params', action='store_true')
    return parser


def init_arguments():
    """
    define general hyperparams
    """
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--config_file', default='paths.json', type=str, \
        help='Config file path - command line arguments will override those in the file.')
    parser.add_argument('--read_best', action='store_true')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--task', type=str, choices=['ihm', 'los'], default='ihm')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpus', type=int, default=-1, help='number of available GPUs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--test', action='store_true', help='enable to skip training and evaluate train model')
    parser.add_argument('--phase', type=str, choices=['val', 'test'], default='test')

    # paths
    parser.add_argument('--version', type=str, help='version tag')
    parser.add_argument('--graph_dir', type=str, help='path of dir storing graph edge data')
    parser.add_argument('--data_dir', type=str, help='path of dir storing raw node data')
    parser.add_argument('--log_path', type=str, help='path to store model')
    parser.add_argument('--load', type=str, help='path to load model from')

    # data
    parser.add_argument('--ts_mask', action='store_true', help='consider time series mask')
    parser.add_argument('--add_flat', action='store_true', help='concatenate data with flat features.')
    parser.add_argument('--add_diag', action='store_true', help='concatenate data with diag features.')
    parser.add_argument('--flat_first', action='store_true', help='concatenate inputs with flat features.')
    parser.add_argument('--random_g', action='store_true', help='use random graph')
    parser.add_argument('--sample_layers', type=int, help='no. of layers for neighbourhood sampling')

    # model
    parser.add_argument('--flat_nhid', type=int, default=64)
    parser.add_argument('--model', type=str, choices=['lstm', 'lstmgnn', 'gnn'], default='lstmgnn')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fc_dim', type=int, default=32)
    parser.add_argument('--main_dropout', type=float, default=0.45)
    parser.add_argument('--main_act_fn', type=str, default='relu')
    parser.add_argument('--batch_norm_loc', type=str, \
        choices=['gnn', 'cat', 'fc'], help='apply batch norm before the specified component.')

    # training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--l2', default=5e-4, type=float, help='5e-4')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--sch', type=str, choices=['cosine', 'plateau'], default='plateau')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--clip_grad', type=float, default=0, help='clipping gradient')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--auto_lr', action='store_true')
    parser.add_argument('--auto_bsz', action='store_true')
    return parser


def init_lstm_args():
    """
    define LSTM-related hyperparams
    """
    parser = init_arguments()

    # shared
    parser.add_argument('--lstm_indim', type=int)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_nhid', type=int, default=64)
    parser.add_argument('--lstm_pooling', type=str, choices=['all', 'last', 'mean', 'max'], default='last')
    parser.add_argument('--lstm_dropout', type=float, default=0.2)
    parser.add_argument('--bilstm', action='store_true')
    return parser


def init_gnn_args(parser):
    """
    define GNN-related hyperparams
    """
    parser.add_argument('--dynamic_g', action='store_true', help='dynamic graph')
    parser.add_argument('--edge_weight', action='store_true', help='use edge weight')
    parser.add_argument('--g_version', type=str, default='default')
    parser.add_argument('--ns_size1', type=int, default=25)
    parser.add_argument('--ns_size2', type=int, default=10)
    parser.add_argument('--gnn_name', type=str, choices=['mpnn', 'sgc', 'gcn', 'gat', 'sage'], default='gat')
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--inductive', action='store_true', help='inductive = train / val /test graphs are different')
    parser.add_argument('--self_loop', action='store_true', help='add self loops')
    
    parser.add_argument('--diag_to_gnn', action='store_true', help='give diag vector to gnn')
    # shared
    parser.add_argument('--gnn_indim', type=int)
    parser.add_argument('--gnn_outdim', type=int, default=64)
    parser.add_argument('--dg_k', type=int, default=3, help='dynamic graph knn')

    # sgc
    parser.add_argument('--sgc_layers', type=int, default=1)
    parser.add_argument('--sgc_k', type=int, default=2)
    parser.add_argument('--no_sgc_bias', action='store_true')
    parser.add_argument('--sgc_norm', type=str)
    
    # gcn
    parser.add_argument('--gcn_nhid', type=int, default=64)
    parser.add_argument('--gcn_layers', type=int, default=1)
    parser.add_argument('--gcn_activation', type=str, default='relu')
    parser.add_argument('--gcn_dropout', type=float, default=0.5)

    # gat
    parser.add_argument('--gat_nhid', type=int, default=64)
    parser.add_argument('--gat_layers', type=int, default=1)
    parser.add_argument('--gat_n_heads', type=int, default=8)
    parser.add_argument('--gat_n_out_heads', type=int, default=8)
    parser.add_argument('--gat_activation', type=str, default='elu')
    parser.add_argument('--gat_featdrop', type=float, default=0.6)
    parser.add_argument('--gat_attndrop', type=float, default=0.6)
    parser.add_argument('--gat_negslope', type=float, default=0.2)
    parser.add_argument('--gat_residual', action='store_true')
    
    # sage
    parser.add_argument('--sage_nhid', type=int, default=64)
    # mpnn
    parser.add_argument('--mpnn_nhid', type=int, default=64)
    parser.add_argument('--mpnn_step_mp', type=int, default=3)
    parser.add_argument('--mpnn_step_s2s', type=int, default=6)
    parser.add_argument('--mpnn_layer_s2s', type=int, default=3)
    return parser


def init_lstmgnn_args():
    """
    define hyperparams for models with LSTM & GNN components (i.e. LSTM-GNNs & dynamic LSTM-GNNs)
    """
    parser = init_lstm_args()
    parser = init_gnn_args(parser)
    parser.add_argument('--lg_alpha', type=float, default=1)
    return parser


def get_lstm_out_dim(config):
    """
    calculate output dimension of lstm
    """
    lstm_last_ts_dim = config['lstm_nhid']
    if config['lstm_pooling'] == 'all':
        lstm_out_dim = config['lstm_nhid'] * 24
    else:
        lstm_out_dim = config['lstm_nhid']
    return lstm_out_dim, lstm_last_ts_dim



def get_version_name(config):
    """
    return str for model version name
    """
    if config['read_best']:
        config['version'] = None
        config['verbose'] = True

    else:
        if config['add_flat']:
            fv = 'flat' + str(config['flat_nhid']) + '_'
        else:
            fv = ''        

        if 'lstm' in config['model']:
            lstm_nm = 'LSTM'
            if config['bilstm']:
                lstm_nm = 'bi' + lstm_nm
            lstm_nm += str(config['lstm_nhid'])
        else:
            lstm_nm = ''
        
        if 'gnn' in config['model']:
            gnn_nm = config['gnn_name'] + str(config[config['gnn_name'] + '_nhid']) + 'out' + str(config['gnn_outdim'])
        else:
            gnn_nm = ''
            
        if config['version'] is None:
            # first about the model
            version = 'e{}{}{}'.format(config['epochs'], lstm_nm, gnn_nm)
            # then about the data
            version += fv
            # finally about training
            version += 'lr' + str(config['lr']) + ('cw_' if config['class_weights'] else '') + ('cos' if config['sch'] == 'cosine' else '')
            version += 'l2' + str(config['l2']) 
            if config['ns_sizes'] != '25_10':
                version += 'ns' + config['ns_sizes'].replace('_', ':')
            if config['tag'] is not None:
                version += 'tag_' + config['tag']
            config['version'] = version
        
    return config




def add_configs(config):
    """
    add in additional configs
    """
    config = vars(config)

    config['verbose'] = False
    config['ns_sizes'] = str(config['ns_size1'] + config['ns_size2']) + '_' + str(config['ns_size1'])
    config['flat_after'] = config['add_flat'] and (not config['flat_first'])
    config['read_lstm_emb'] = False

    if config['add_diag']:
        assert config['add_flat'] # otherwise need to change ts_reader (collect_ts_flat_labels fn)

    if 'gnn' in config['model']:
        if not config['dynamic_g']:
            assert (config['random_g']) or (config['g_version'] is not None)

    # task
    if config['task'] == 'ihm':
        config['classification'] = True
        config['out_dim'] = 2
        config['num_cls'] = 2
        config['final_act_fn'] = None
    else:
        config['classification'] = False
        config['out_dim'] = 1
        config['num_cls'] = 1
        config['final_act_fn'] = 'hardtanh'

    config['lstm_attn_type'] = None

    # model dimensions
    config['lstm_outdim'], config['lstm_last_ts_dim'] = get_lstm_out_dim(config)

    if config['model'] == 'lstmgnn':
        config['gnn_indim'] = config['lstm_outdim']
        config['add_last_ts'] = True # true by default
    else:
        config['add_last_ts'] = False
    
    if 'gnn' in config['model']:
        if config['gnn_name'] == 'gat':
            config['gat_heads'] = ([config['gat_n_heads']] * config['gat_layers']) + [config['gat_n_out_heads']]
        if not (config['add_flat'] and (not config['flat_first'])): # i.e directly output class after gnn
            config['gnn_outdim'] = config['out_dim']
    if (config['model'] == 'lstmgnn') and (not config['dynamic_g']):
        if config['gnn_name'] == 'mpnn':
            config['ns_sizes'] = str(config['ns_size1'])

    elif config['model'] == 'lstm':
        config['sampling_layers'] = 1


    # training details
    if config['cpu']:
        num_gpus = 0
        config['gpus'] = None
    else:
        if config['gpus'] is not None:
            num_gpus = torch.cuda.device_count() if config['gpus'] == -1 else config['gpus']
            if num_gpus > 0:
                config['batch_size'] *= num_gpus
                config['num_workers'] *= num_gpus
        else:
            num_gpus = 0
    config['num_gpus'] = num_gpus
    config['multi_gpu'] = num_gpus > 1

    if 'config_file' in config:
        read_params_from_file(config)

    if config['read_best']:
        add_best_params(config)

    # define log path
    if config['model'] == 'gnn':
        dir_name =  config['gnn_name'] + '_' + config['task']
        if 'whole_g' in config:
             dir_name = 'whole_' + dir_name
    elif config['model'] == 'lstmgnn':
        dir_name = 'lstm' + config['gnn_name'] + '_alpha' + str(config['lg_alpha'])
    else:
        dir_name = config['model']

    inputs = ''
    if config['ts_mask']:
        inputs += 'tm_'
    if config['add_flat']:
        inputs += 'flat'
        if config['flat_first']:
            inputs += 'f'
        if config['add_diag']:
            inputs += 'd'
    if inputs == '':
        inputs = 'seq'
    
    if config['read_best'] and 'repeat_path' in config:
        config['log_path'] = config['repeat_path']
    
    if 'gnn' in config['model']:
        graph_v = 'graphV' + str(config['g_version'])
        if config['dynamic_g']:
            config['log_path'] = Path(config['log_path']) / config['task'] / inputs / (config['model'] + '_led') / dir_name
        else:
            dir_name += '_' + graph_v
            if config['load'] is not None:
                config['log_path'] = Path(config['log_path']) / config['task'] / inputs / (config['model'] + '_pt') / dir_name
            else:    
                config['log_path'] = Path(config['log_path']) / config['task'] / inputs / config['model'] / dir_name
    else:
        config['log_path'] = Path(config['log_path']) / config['task'] / inputs / 'lstm_baselines'

    get_version_name(config)

    print('version name = ', config['version'])

    return config


def read_params_from_file(arg_dict, overwrite=False):
    """
    Read params defined in config_file (paths.py by default.)
    """
    if '/' not in arg_dict['config_file']:
        config_path = Path(sys.path[0]) /  arg_dict['config_file']
    else:
        config_path = Path(arg_dict['config_file'])
    
    data = load_json(config_path)
    arg_dict.pop('config_file')

    if not overwrite:
        for key, value in data.items():
            if isinstance(value, list) and (key in arg_dict):
                for v in value:
                    arg_dict[key].append(v)
            elif (key not in arg_dict) or (arg_dict[key] is None):
                arg_dict[key] = value
    else:
        for key, value in data.items():
            arg_dict[key] = value

