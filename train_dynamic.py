"""
main file for training Dynamic LSTM-GNNs
"""
import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs
from src.dataloader.ts_reader import LstmDataset, collate_fn
from src.models.dgnn import DynamicLstmGnn
from src.models.utils import get_checkpoint_path, seed_everything
from src.metrics import get_loss_function, get_metrics, get_per_node_result
from src.args import init_lstmgnn_args, add_configs
from src.utils import write_json, write_pkl
from src.utils import record_results


class DynamicGraphModel(pl.LightningModule):
    """
    Dynamic Graph Model:
    """
    def __init__(self, config, collate, train_set=None, val_set=None, test_set=None):
        super().__init__()
        self.config = config
        self.trainset = train_set
        self.validset = val_set
        self.testset = test_set
        self.learning_rate = self.config['lr']
        self.collate = collate
        self.task = config['task']
        self.is_cls = config['classification']
        self.net = DynamicLstmGnn(config)

        self.loss = get_loss_function(self.task, config['class_weights'])
        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda truth, pred : get_metrics(truth, pred, config['verbose'], config['classification'])
        self.per_node_metrics = lambda truth, pred : get_per_node_result(truth, pred, self.testset.idx_test, config['classification'])
        self.lg_alpha = config['lg_alpha']

    def on_train_start(self):
        seed_everything(self.config['seed'])
    
    def forward(self, seq, flat):
        out = self.net(seq, flat=flat)
        return out
    
    def add_losses(self, out, out_lstm, bsz_y):
        train_loss = self.loss(out.squeeze(), bsz_y)
        train_loss_lstm = self.loss(out_lstm.squeeze(), bsz_y)
        tot_loss = train_loss + train_loss_lstm * self.lg_alpha
        return train_loss, train_loss_lstm, tot_loss

    def training_step(self, batch, batch_idx):
        inputs, truth, _ = batch
        seq, flat = inputs
        pred, pred_lstm = self(seq, flat)
        train_loss, train_loss_lstm, tot_loss = self.add_losses(pred, pred_lstm, truth)
        log_dict = {'train_loss': train_loss, 'train_loss_lstm': train_loss_lstm, 'train_tot_loss': tot_loss}
        return {'loss': train_loss, 'log': log_dict}#, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        inputs, truth, _ = batch
        seq, flat = inputs
        out, out_lstm = self(seq, flat)
        loss, loss_lstm, tot_loss = self.add_losses(out, out_lstm, truth)
        results = {
            'val_loss': loss,
            'val_loss_lstm': loss_lstm,
            'val_tot_loss': tot_loss,
            'truth': truth, 'pred': out, 'pred_lstm': out_lstm}
        return results

    def test_step(self, batch, batch_idx):
        inputs, truth, ids = batch
        seq, flat = inputs
        out, out_lstm = self(seq, flat)
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
        per_node = self.per_node_metrics(collect_dict['truth'], collect_dict['pred'])
        results = {'log': log_dict, 'progress_bar': losses}
        results = {**results, **log_dict}
        return results

    def test_epoch_end(self, outputs):
        collect_dict = self.collect_outputs(outputs)
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
        return DataLoader(self.trainset, collate_fn=self.collate, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, collate_fn=self.collate, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, collate_fn=self.collate, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=False)

    @staticmethod
    def load_model(log_dir, **hconfig):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param config: list of named arguments, used to update the model hyperparameters
        """
        assert os.path.exists(log_dir)
        # load hparams
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            config = yaml.load(fp, Loader=yaml.Loader)
            config.update(hconfig)

        loaderDict, collate = get_data(config)

        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        print(f'Loading model {model_path.parent.stem}')
        args = {'config': dict(config), 'collate': collate,
            'train_set': loaderDict['train'], 'val_set': loaderDict['val'], 'test_set': loaderDict['test']}
        model = DynamicGraphModel.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, loaderDict, collate


def get_data(config):
    """
    produce dataloaders for training and validating
    """
    loaderDict = {split: LstmDataset(config, split) for split in ['train', 'val', 'test']}
    config['lstm_indim'] = loaderDict['train'].ts_dim
    config['num_flat_feats'] = loaderDict['train'].flat_dim if config['add_flat'] else 0
    config['class_weights'] = loaderDict['train'].class_weights if config['class_weights'] else False
    collate = lambda x: collate_fn(x, config['task'])

    return loaderDict, collate


def main(config):
    """
    Main function for training Dyanmic LSTM-GNNs.
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    loaderDict, collate = get_data(config)

    # define logger
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    # define model
    if config['debug']:
        model = DynamicGraphModel(config, collate, loaderDict['val'], loaderDict['test'], loaderDict['test'])
    else:    
        model = DynamicGraphModel(config, collate, loaderDict['train'], loaderDict['val'], loaderDict['test'])
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
            ret = trainer.test(test_dataloaders=model.val_dataloader())
        else:
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

    model, config, loaderDict, collate = DynamicGraphModel.load_model(log_dir, \
        data_dir=hparams['data_dir'], 
        multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    # Evaluate the model
    test_dataloader = DataLoader(loaderDict[phase], collate_fn=collate, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    test_results = trainer.test(model, test_dataloaders=test_dataloader)
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
    config.model = 'lstmgnn'
    config.dynamic_g = True
    config = add_configs(config)
    
    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['test']:
        main_test(config)
    else:
        main(config)