"""
main file for training LSTM-GNNs
"""
import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs
from src.dataloader.pyg_reader import GraphDataset
from src.models.pyg_lstmgnn import NsLstmGNN
from src.models.utils import get_checkpoint_path, seed_everything
from src.metrics import get_loss_function, get_metrics, get_per_node_result
from src.args import add_configs, init_lstmgnn_args
from src.utils import write_json, write_pkl
from torch_geometric.data import NeighborSampler
from src.dataloader.ts_reader import LstmDataset, collate_fn
from src.utils import record_results


class Model(pl.LightningModule):
    """
    Basic GNN model
    """
    def __init__(self, config, dataset, train_loader, subgraph_loader, eval_split='test', \
            chkpt=None, get_emb=False, get_logits=False, get_attn=False):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.batch_loader = train_loader
        self.subgraph_loader = subgraph_loader
        self.learning_rate = config['lr']
        self.get_emb = get_emb
        self.get_logits = get_logits
        self.get_attn = get_attn

        self.net = NsLstmGNN(self.config)
        
        if chkpt is not None:
            self.net.load_state_dict(chkpt['state_dict'], strict=False)
            print('Loaded states')
                
        self.loss = get_loss_function(config['task'], config['class_weights'])

        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda truth, pred : get_metrics(truth, pred, config['verbose'], config['classification'])
        self.per_node_metrics = lambda truth, pred : get_per_node_result(truth, pred, self.dataset.idx_test, config['classification'])

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'test' else self.dataset.data.test_mask

        entire_set = LstmDataset(config)
        collate = lambda x: collate_fn(x, config['task'])
        self.ts_loader = DataLoader(entire_set, collate_fn=collate, \
                batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
        self.lg_alpha = config['lg_alpha']

    def on_train_start(self):
        seed_everything(self.config['seed'])
    
    def forward(self, x, flat, adjs, batch_size, edge_weight):
        out, out_lstm = self.net(x, flat, adjs, batch_size, edge_weight)
        return out, out_lstm

    def add_losses(self, out, out_lstm, bsz_y):
        train_loss = self.loss(out.squeeze(), bsz_y)
        train_loss_lstm = self.loss(out_lstm.squeeze(), bsz_y)
        tot_loss = train_loss + train_loss_lstm * self.lg_alpha
        return train_loss, train_loss_lstm, tot_loss

    def training_step(self, batch, batch_idx):
        # these are train-masked already (from train-dataloader)
        batch_size, n_id, adjs = batch
        in_x = self.dataset.data.x[n_id].to(self.device)
        in_flat = self.dataset.data.flat[n_id[:batch_size]].to(self.device)
        edge_weight = self.dataset.data.edge_attr.to(self.device)
        bsz_y = self.dataset.data.y[n_id[:batch_size]].to(self.device)
        out, out_lstm = self(in_x, in_flat, adjs, batch_size, edge_weight)
        train_loss, train_loss_lstm, tot_loss = self.add_losses(out, out_lstm, bsz_y)
        log_dict = {'train_loss': train_loss, 'train_loss_lstm': train_loss_lstm, 'train_tot_loss': tot_loss}
        return {'loss': tot_loss, 'log': log_dict}#, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        # there's just one step for the validation step:
        # - we're not using batch / batch_idx (it's just dummy)
        x = self.dataset.data.x.to(self.device)
        flat = self.dataset.data.flat.to(self.device)
        edge_weight = self.dataset.data.edge_attr.to(self.device)
        truth = self.dataset.data.y
        out, out_lstm = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device) # within this - loop the entire subgraph loader
        truth = truth[self.dataset.data.val_mask].to(self.device)
        out = out[self.dataset.data.val_mask]
        out_lstm = out_lstm[self.dataset.data.val_mask]
        loss, loss_lstm, tot_loss = self.add_losses(out, out_lstm, truth)
        results = {
            'val_loss': loss,
            'val_loss_lstm': loss_lstm,
            'val_tot_loss': tot_loss,
            'truth': truth, 'pred': out, 'pred_lstm': out_lstm}
        return results

    def test_step(self, batch, batch_idx):
        # there's just one step for the test step:
        # - we're not using batch / batch_idx (it's just dummy)
        if self.get_emb or self.get_logits:
            x = self.dataset.data.x.to(self.device)
            flat = self.dataset.data.flat.to(self.device)
            edge_weight = self.dataset.data.edge_attr.to(self.device)
            truth = self.dataset.data.y
            if self.get_attn:
                edge_index = self.dataset.data.edge_index
                out, out_lstm, edge_index_w_self_loops, all_edge_attn = self.net.inference_w_attn(x, flat, edge_weight, edge_index, self.ts_loader, self.subgraph_loader, self.device)
                # out, out_lstm, edge_index_w_self_loops, all_edge_attn = self.net.inference_w_attn(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device)
                results = {'hid': out, 'edge_index': edge_index_w_self_loops, 'edge_attn_1': all_edge_attn[0], 'edge_attn_2': all_edge_attn[1]}
            else:
                if self.get_logits:
                    out, out_lstm = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device) # within this - loop the entire subgraph loader
                else:
                    out, out_lstm = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device, get_emb=True)
                results = {'hid': out}
        else:
            x = self.dataset.data.x.to(self.device)
            flat = self.dataset.data.flat.to(self.device)
            edge_weight = self.dataset.data.edge_attr.to(self.device)
            truth = self.dataset.data.y
            out, out_lstm = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device) # within this - loop the entire subgraph loader
            truth = truth[self.eval_mask].to(self.device)
            out = out[self.eval_mask]
            out_lstm = out_lstm[self.eval_mask]
            loss, loss_lstm, tot_loss = self.add_losses(out, out_lstm, truth)
            results = {
                'test_loss': loss,
                'test_loss_lstm': loss_lstm,
                'test_tot_loss': tot_loss,
                'truth': truth, 'pred': out, 'pred_lstm': out_lstm}
        return results

    def validation_epoch_end(self, outputs):
        collect_dict = self.collect_outputs(outputs)
        log_dict_1 = self.compute_metrics(collect_dict['truth'], collect_dict['pred'])
        log_dict_2 = self.compute_metrics(collect_dict['truth'], collect_dict['pred_lstm'])
        log_dict_2 = {n+ '_lstm': log_dict_2[n] for n in log_dict_2}
        losses = {n: float(collect_dict[n]) for n in ['val_loss', 'val_loss_lstm', 'val_tot_loss']}
        log_dict = {**log_dict_1, **log_dict_2, **losses}
        # per_node = self.per_node_metrics(collect_dict)
        results = {'log': log_dict}#, 'progress_bar': losses}
        results = {**results, **log_dict}
        return results

    def test_epoch_end(self, outputs):
        collect_dict = self.collect_outputs(outputs)
        if self.get_emb or self.get_logits or self.get_attn:
            results = collect_dict
        else:
            log_dict_1 = self.compute_metrics(collect_dict['truth'], collect_dict['pred'])
            log_dict_2 = self.compute_metrics(collect_dict['truth'], collect_dict['pred_lstm'])
            log_dict_2 = {n+ '_lstm': log_dict_2[n] for n in log_dict_2}
            log_dict = {**log_dict_1, **log_dict_2}
            log_dict = {'test_' + m: log_dict[m] for m in log_dict}
            losses = {n: float(collect_dict[n]) for n in ['test_loss', 'test_loss_lstm', 'test_tot_loss']}
            log_dict = {**log_dict, **losses}

            results = {'log': log_dict}
            per_node_1 = self.per_node_metrics(collect_dict['truth'], collect_dict['pred'])
            per_node_2 = self.per_node_metrics(collect_dict['truth'], collect_dict['pred_lstm'])
            per_node_2 = {n+ '_lstm': per_node_2[n] for n in per_node_2}
            per_node = {**per_node_1, **per_node_2}
            results['per_node'] = per_node
            results = {**results, **log_dict}
        return results

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.config['l2'])
        if self.config['sch'] == 'cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        else:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        return [opt], [sch]
    
    def train_dataloader(self):
        return self.batch_loader

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

    @staticmethod
    def load_model(log_dir, **hparams):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param config: list of named arguments, used to update the model hyperparameters
        """
        assert os.path.exists(log_dir)
        # load hparams
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            config = yaml.load(fp, Loader=yaml.Loader)
            config.update(hparams)

        dataset, train_loader, subgraph_loader = get_data(config)

        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        print(f'Loading model {model_path.parent.stem}')
        args = {'config': dict(config), 'dataset': dataset, \
            'train_loader': train_loader, 'subgraph_loader': subgraph_loader}
        model = Model.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, dataset, train_loader, subgraph_loader


def get_data(config, us=None, vs=None):
    """
    produce dataloaders for training and validating
    """
    dataset = GraphDataset(config, us, vs)
    config['lstm_indim'] = dataset.x_dim
    config['num_flat_feats'] = dataset.flat_dim
    config['class_weights'] = dataset.class_weights
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    sample_sizes = [config['ns_size1']] if config['gnn_name'] == 'mpnn' else [config['ns_size1'] + config['ns_size2'], config['ns_size1']]

    # train loader - only samples from the train nodes
    train_loader = NeighborSampler(dataset.data.edge_index, node_idx=dataset.data.train_mask,
                               sizes=sample_sizes, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    # val / test loader - samples from the entire graph
    subgraph_loader = NeighborSampler(dataset.data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    return dataset, train_loader, subgraph_loader


def main(config):
    """
    Main function for training LSTM-GNN.
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    dataset, train_loader, subgraph_loader = get_data(config)

    # define logger
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    # load from lstm:
    chkpt = None
    if config['load'] is not None:
        chkpt_path = get_checkpoint_path(config['load'])
        chkpt = torch.load(chkpt_path, map_location=torch.device('cpu'))

     # define model
    model = Model(config, dataset, train_loader, subgraph_loader, chkpt=chkpt)
    
    trainer = pl.Trainer(
        gpus=config['gpus'],
        logger=logger,
        max_epochs=config['epochs'],
        distributed_backend='dp',
        precision=16 if config['use_amp'] else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        # resume_from_checkpoint=chkpt,
        auto_lr_find=config['auto_lr'],
        auto_scale_batch_size=config['auto_bsz']
    )
    trainer.fit(model)

    for phase in ['test', 'valid']:
        if phase == 'valid':
            trainer.eval_split = 'val'
            trainer.eval_mask = dataset.data.val_mask
            print(phase, trainer.eval_split)
        
        ret = trainer.test()
        if isinstance(ret, list):
            ret = ret[0]

        per_node = ret.pop('per_node')
        test_results = ret
        res_dir = Path(config['log_path']) / 'default' 
        if config['version'] is not None:
            res_dir = res_dir / config['version']
        else:
            res_dir = res_dir / ('results_' + str(config['seed']))
        print(phase, ':', test_results)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')

        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)



def main_forward_pass(hparams):
    """
    Main function to load a trained model and execute a forward pass to get logits, embeddings and attention weights for analysis.
    """
    log_dir = hparams['load']
    model, config, dataset, train_loader, subgraph_loader = Model.load_model(log_dir, \
            data_dir=hparams['data_dir'], graph_dir=hparams['graph_dir'], \
            multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    if hparams['fp_emb']:
        model.get_emb = True
    else:
        model.get_logits = True
        if hparams['fp_attn']:
            model.get_attn = True

    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    import numpy as np
    test_results = trainer.test(model)
    if isinstance(test_results, list):
        test_results = test_results[0]

    # logits / hidden vector
    hid = test_results['hid']
    name = 'lstm' + config['gnn_name'] + ('_embeddings.npy' if hparams['fp_emb'] else '_logits.npy')
    out_path = Path(log_dir) / name
    with open(out_path, 'wb') as f:
        np.save(f, hid)
    print('saved at', out_path)

    # get attentions:
    if hparams['fp_attn']:
        for item in ['edge_index', 'edge_attn_1', 'edge_attn_2']:
            out_path = Path(log_dir) / ('lstm' + config['gnn_name'] + '_' + item + '.npy')
            dat = test_results[item]
            with open(out_path, 'wb') as f:
                np.save(f, dat)
            print('saved at', out_path)



def main_test(hparams, path_results=None):
    """
    main function to load and evaluate a trained model. 
    """
    assert (hparams['load'] is not None) and (hparams['phase'] is not None)
    phase = hparams['phase']
    log_dir = hparams['load']

    # Load trained model
    print(f'Loading from {log_dir} to evaluate {phase} data.')

    model, config, dataset, train_loader, subgraph_loader = Model.load_model(log_dir, data_dir=hparams['data_dir'], graph_dir=hparams['graph_dir'], \
        multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    # Evaluate the model
    if phase == 'valid':
        trainer.eval_split = 'val'
        trainer.eval_mask = dataset.data.val_mask
        print(phase, trainer.eval_split)
    
    test_results = trainer.test(model)
    if isinstance(test_results, list):
        test_results = test_results[0]
    per_node = test_results.pop('per_node')
    print(phase, ':', test_results)
    # Save evaluation results
    results_path = Path(log_dir) / f'{phase}_results.json'
    write_json(test_results, results_path, sort_keys=True, verbose=True)
    write_pkl(per_node, Path(log_dir) / f'{phase}_per_node.pkl')

    if path_results is None:
        path_results = Path(log_dir).parent / 'results.csv'
    tmp = {'version': hparams['version']}
    tmp = {**tmp, **config}
    record_results(path_results, tmp, test_results)


if __name__ == '__main__':
    # define configs
    parser = init_lstmgnn_args()
    parser.add_argument('--fp_emb', action='store_true', help='forward pass to get embeddings')
    parser.add_argument('--fp_logits', action='store_true', help='forward pass to get logits')
    parser.add_argument('--fp_attn', action='store_true', help='forward pass to get attention weights (for GAT)')
    config = parser.parse_args()
    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    
    if config['fp_emb'] or config['fp_logits'] or config['fp_attn']:
        main_forward_pass(config)
    
    if config['test']:
        main_test(config)
    
    else:
        main(config)


