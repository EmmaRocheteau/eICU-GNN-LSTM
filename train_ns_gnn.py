"""
main file for training GNNs (with node sampling)
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
from src.models.pyg_ns import NsGNN
from src.models.utils import get_checkpoint_path, seed_everything
from src.metrics import get_loss_function, get_metrics, get_per_node_result
from src.args import add_configs, init_lstmgnn_args
from src.utils import write_json, write_pkl
from torch_geometric.data import NeighborSampler
from src.utils import record_results


class NsGnnModel(pl.LightningModule):
    """
    Basic GNN model
    """
    def __init__(self, config, dataset, train_loader, subgraph_loader, eval_split='test'):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.batch_loader = train_loader
        self.subgraph_loader = subgraph_loader
        self.learning_rate = config['lr']

        self.net = NsGNN(config)

        self.loss = get_loss_function(config['task'], config['class_weights'])

        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda x: get_metrics(x['truth'], x['pred'], config['verbose'], config['classification'])
        self.per_node_metrics = lambda x: get_per_node_result(x['truth'], x['pred'], self.dataset.idx_test, config['classification'])

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'val' else self.dataset.data.test_mask


    def on_train_start(self):
        seed_everything(self.config['seed'])

    def forward(self, x, flat, adjs, edge_weight):
        out = self.net(x, flat, adjs, edge_weight)
        return out

    def training_step(self, batch, batch_idx):
        # these are train-masked already (from train-dataloader)
        batch_size, n_id, adjs = batch
        in_x = self.dataset.data.x[n_id].to(self.device)
        in_flat = self.dataset.data.flat[n_id[:batch_size]].to(self.device)
        bsz_y = self.dataset.data.y[n_id[:batch_size]].to(self.device)
        edge_weight = self.dataset.data.edge_attr.to(self.device)
        out = self(in_x, in_flat, adjs, edge_weight)
        train_loss = self.loss(out.squeeze(), bsz_y)
        log_dict = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': log_dict}

    def validation_step(self, batch, batch_idx):
        """
        There's just one step for the test step, and we're not using batch / batch_idx (dummy)
        """
        x = self.dataset.data.x.to(self.device)
        flat = self.dataset.data.flat.to(self.device)
        truth = self.dataset.data.y
        edge_weight = self.dataset.data.edge_attr.to(self.device)
        out = self.net.inference(x, flat, self.subgraph_loader, self.device, edge_weight) # within this - loop the entire subgraph loader
        truth = truth[self.dataset.data.val_mask].to(self.device)
        pred = out[self.dataset.data.val_mask].to(self.device)
        results = {'val_loss': self.loss(pred.squeeze(), truth), 'truth': truth, 'pred': pred}
        return results

    def test_step(self, batch, batch_idx):
        """
        There's just one step for the test step, and we're not using batch / batch_idx (dummy)
        """
        x = self.dataset.data.x.to(self.device)
        flat = self.dataset.data.flat.to(self.device)
        truth = self.dataset.data.y
        edge_weight = self.dataset.data.edge_attr.to(self.device)
        out = self.net.inference(x, flat, self.subgraph_loader, self.device, edge_weight)
        truth = truth[self.eval_mask].to(self.device)
        pred = out[self.eval_mask]
        results = {'test_loss': self.loss(pred.squeeze(), truth), 'truth': truth, 'pred': pred}
        return results

    def validation_epoch_end(self, outputs):
        collect_dict = self.collect_outputs(outputs)
        log_dict = self.compute_metrics(collect_dict)
        log_dict['val_loss'] = float(collect_dict['val_loss'])
        results = {'log': log_dict}
        results = {**results, **log_dict}
        return results

    def test_epoch_end(self, outputs):
        collect_dict = self.collect_outputs(outputs)
        log_dict = self.compute_metrics(collect_dict)
        log_dict = {'test_' + m: log_dict[m] for m in log_dict}
        log_dict['test_loss'] = float(collect_dict['test_loss'])
        results = {'log': log_dict}

        results['per_node'] = self.per_node_metrics(collect_dict)
        results = {**results, **log_dict}
        return results

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.config['l2'])
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
        model = NsGnnModel.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, dataset, train_loader, subgraph_loader


def get_data(config, us=None, vs=None):
    """
    produce dataloaders for training and validating
    """
    dataset = GraphDataset(config, us, vs)
    config['gnn_indim'] = dataset.x_dim
    config['num_flat_feats'] = dataset.flat_dim
    config['class_weights'] = dataset.class_weights
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    if config['gnn_name'] != 'mpnn':
        sample_sizes = [config['ns_size1'] + config['ns_size2'], config['ns_size1']]
    else:
        sample_sizes = [config['ns_size1']]
    train_loader = NeighborSampler(dataset.data.edge_index, node_idx=dataset.data.train_mask,
                               sizes=sample_sizes, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    subgraph_loader = NeighborSampler(dataset.data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    return dataset, train_loader, subgraph_loader


def main(config):
    """
    Main function for training GNNs (with node sampling).
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    dataset, train_loader, subgraph_loader = get_data(config)

    # define logger
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    # define model
    model = NsGnnModel(config, dataset, train_loader, subgraph_loader)
    chkpt = None if config['load'] is None else get_checkpoint_path(config['load'])

    trainer = pl.Trainer(
        gpus=config['gpus'],
        logger=logger,
        max_epochs=config['epochs'],
        distributed_backend='dp',
        precision=16 if config['use_amp'] else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        resume_from_checkpoint=chkpt,
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



def main_test(hparams, path_results=None):
    """
    main function to load and evaluate a trained model. 
    """
    assert (hparams['load'] is not None) and (hparams['phase'] is not None)
    phase = hparams['phase']
    log_dir = hparams['load']

    # Load trained model
    print(f'Loading from {log_dir} to evaluate {phase} data.')

    model, config, dataset, train_loader, subgraph_loader = NsGnnModel.load_model(log_dir, \
        data_dir=hparams['data_dir'], graph_dir=hparams['graph_dir'], \
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
    config = parser.parse_args()
    config.model = 'gnn'
    config.flatten = True

    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['test']:
        main_test(config)
    
    else:
        main(config)