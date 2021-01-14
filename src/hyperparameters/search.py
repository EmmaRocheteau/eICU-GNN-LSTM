from pytorch_lightning.callbacks import Callback
from ray import tune
from ray.tune import CLIReporter
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from src.hyperparameters import ns_gnn_2d, ns_gnn_4, dynamic, lstmgnn


class ihm_TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics['val_loss'],
            acc=trainer.callback_metrics['acc'],
            auroc=trainer.callback_metrics['auroc'],
            auprc=trainer.callback_metrics['auprc'])


class los_TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics['val_loss'],
            mad=trainer.callback_metrics['mad'],
            mse=trainer.callback_metrics['mse'],
            mape=trainer.callback_metrics['mape'],
            msle=trainer.callback_metrics['msle'],
            r2=trainer.callback_metrics['r2'],
            kappa=trainer.callback_metrics['kappa'])


all_grid = {
    'batch_size': tune.choice([32, 64, 128]),
    'lr': tune.loguniform(5e-4, 1e-3),
    'l2': tune.loguniform(1e-5, 1e-3),
    'main_dropout': tune.uniform(0, 0.5)
}

nsgnn_grid = {
    "lg_alpha": tune.loguniform(0.5, 3),
}


gnn_specific_grid = {
    'sage': 
        {   
            'ns_size1': tune.choice([5, 10, 15, 20, 30]),
            'ns_size2': tune.choice([5, 10, 15, 20, 30]),
            'gnn_outdim': tune.choice([64, 128, 256, 512]),
            'sage_nhid': tune.choice([64, 128, 256, 512])
        },
    'gat':
        {      
            'ns_size1': tune.choice([5, 10, 15, 20, 30]),
            'ns_size2': tune.choice([5, 10, 15, 20, 30]),
            'gnn_outdim': tune.choice([64, 128, 256, 512]),
            'gat_nhid': tune.choice([64, 128, 256, 512]),
            'gat_n_heads': tune.choice([8, 10, 12]),
            'gat_n_out_heads': tune.choice([6, 8, 10]),
            'gat_featdrop': tune.uniform(0.2, 0.7),
            'gat_attndrop': tune.uniform(0.2, 0.7),
        },
    'gcn':
        {   
            'ns_size1': tune.choice([5, 10, 15, 20, 30]),
            'ns_size2': tune.choice([5, 10, 15, 20, 30]),
            'gnn_outdim': tune.choice([64, 128, 256, 512]),
            'gcn_nhid': tune.choice([64, 128, 256, 512]),
            'gcn_dropout': tune.uniform(0.2, 0.7)
        },
    'mpnn':
        {   
            'ns_size1': tune.choice([10, 15, 20, 30]),
            'gnn_outdim': tune.choice([64, 128, 256, 512]),
            'mpnn_nhid': tune.choice([64, 128, 256, 512]),
            'mpnn_step_mp': tune.choice([1, 2, 3, 4, 5])
        }
    }




def main_tune(tune_function, config):
    parameter_columns = ['batch_size', 'lr', 'l2', 'main_dropout']

    for key, value in all_grid.items():
        config[key] = value
    
    if config['model'] == 'lstm':
        pass

    elif config['dynamic_g']:
        gnn_grid = gnn_specific_grid[config['gnn_name']]
        for key, value in gnn_grid.items():
            if ('ns_size' not in key):
                config[key] = value
                parameter_columns.append(key)

    elif 'gnn' in config['model']: # lstmgnn / ns_gnn
        for key, value in nsgnn_grid.items():
            config[key] = value
            parameter_columns.append(key)
        gnn_grid = gnn_specific_grid[config['gnn_name']]
        for key, value in gnn_grid.items():
            config[key] = value
            parameter_columns.append(key)

    if config['fix_g_params'] or config['fix_l_params']:
        if config['dynamic_g']:
            best_params = dynamic[config['task']][config['gnn_name']]
        elif config['model'] == 'lstmgnn':
            best_params = lstmgnn[config['task']][config['gnn_name']]
        elif config['model'] == 'gnn':
            if config['g_version'] == '2d':
                best_params = ns_gnn_2d[config['task']][config['gnn_name']]
            elif '4' in config['g_version']:
                best_params = ns_gnn_4[config['task']][config['gnn_name']]
        else:
            print(config['model'], config['dynamic_g'], config['g_version'])
        # fixing the values 
        fixed_params = []

        l_params = ['batch_size', 'lr', 'l2', 'main_dropout', 'lg_alpha'] + [l for l in config if 'drop' in l]
        if config['fix_g_params']:
            for key, fixed_value in best_params.items():
                if key not in l_params:
                    config[key] = fixed_value
                    fixed_params.append(key)
                    print(f'{key}: {fixed_value}')
        else: # fixing learning parameters
            for key, fixed_value in best_params.items():
                if key in l_params:
                    config[key] = fixed_value
                    fixed_params.append(key)
        parameter_columns = [p for p in parameter_columns if p not in fixed_params]
        print('*** using best values for these params', fixed_params)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=config['epochs'],
        grace_period=config['grace_period'],
        reduction_factor=2)

    if config['task'] == 'ihm':
        metric_cols = ['loss' 'acc', 'auroc', 'auprc', 'training_iteration']
    else:
        metric_cols = ['loss', 'mad', 'mse', 'mape', 'msle', 'r2', 'kappa', 'training_iteration']

    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=metric_cols)

    exp_name = config['task'] + '_' + config['model'] + '_' + config['gnn_name']

    tune.run(
        partial(
            tune_function),
        name = exp_name,
        local_dir=config['ray_dir'],
        resources_per_trial={"cpu": config['cpus_per_trial'], "gpu": config['gpus_per_trial']},
        config=config,
        num_samples=config['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter)

