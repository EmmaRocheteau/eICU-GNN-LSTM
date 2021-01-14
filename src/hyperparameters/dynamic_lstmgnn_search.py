import os
os.environ["SLURM_JOB_NAME"] = "bash"
import pytorch_lightning as pl
from train_dynamic import get_data, DynamicGraphModel
from src.hyperparameters.search import ihm_TuneReportCallback, los_TuneReportCallback, main_tune
from src.args import init_lstmgnn_args, add_tune_params, add_configs


def main_train(config):
    loaderDict, collate = get_data(config)

    # define model
    if config['debug']:
        model = DynamicGraphModel(config, collate, loaderDict['val'], loaderDict['test'], loaderDict['test'])
    else:
        model = DynamicGraphModel(config, collate, loaderDict['train'], loaderDict['val'], loaderDict['test'])

    trcb = [ihm_TuneReportCallback()] if config['task'] == 'ihm' else [los_TuneReportCallback()]

    trainer = pl.Trainer(
        gpus=config['gpus'],
        progress_bar_refresh_rate=0,
        weights_summary=None,
        max_epochs=config['epochs'],
        distributed_backend='dp',
        precision=16 if config['use_amp'] else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        callbacks=trcb
    )
    trainer.fit(model)



if __name__ == '__main__':
    parser = init_lstmgnn_args()
    parser = add_tune_params(parser)
    config = parser.parse_args()
    config.model = 'lstmgnn'
    config.dynamic_g = True
    config = add_configs(config)
    main_tune(main_train, config)