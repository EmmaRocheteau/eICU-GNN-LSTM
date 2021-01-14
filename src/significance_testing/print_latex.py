import pandas as pd
import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def print_metrics_los(df):
    mean_all, conf_bound = mean_confidence_interval(df)

    for i, (m, cb) in enumerate(zip(mean_all, conf_bound)):
        if i == 0:
            m = str(np.round(m, 2)).ljust(4, '0')
            cb = str(np.round(cb, 2)).ljust(4, '0')
        elif i in [1, 2]:
            m = str(np.round(m, 1)).ljust(4, '0')
            cb = str(np.round(cb, 1)).ljust(3, '0')
    #    else:
    #        m = str(np.round(m, 3)).ljust(5, '0')
    #        cb = str(np.round(cb, 3)).ljust(5, '0')
        else:
            m = str(np.round(m, 3)).ljust(4, '0')
            cb = str(np.round(cb, 3)).ljust(4, '0')
        and_space = ' & ' if i < 5 else ' \\\\'
        print('\\tiny{}{}$\pm${}{}{}'.format('{', m, cb, '}', and_space), end='', flush=True)

def print_metrics_ihm(df):
    mean_all, conf_bound = mean_confidence_interval(df)

    for i, (m, cb) in enumerate(zip(mean_all, conf_bound)):
        if i == 0:
            m = str(np.round(m, 3)).ljust(4, '0')
            cb = str(np.round(cb, 3)).ljust(4, '0')
        elif i == 1:
            m = str(np.round(m, 3)).ljust(4, '0')
            cb = str(np.round(cb, 3)).ljust(5, '0')
        and_space = ' & ' if i < 5 else ' \\\\'
        print('\\tiny{}{}$\pm${}{}{}'.format('{', m, cb, '}', and_space), end='', flush=True)

def main(los_name=None, ihm_name=None):
    print('Experiment: {} {}'.format(los_name, ihm_name))
    if ihm_name is not None:
        ihm = pd.read_csv('local_log/results/' + ihm_name + '.csv')
        ihm = ihm[['test_auroc', 'test_auprc']]
        print_metrics_ihm(ihm)
    else:
        print('& & ')
    if los_name is not None:
        los = pd.read_csv('local_log/results/' + los_name + '.csv')
        los = los[['test_mad', 'test_mape', 'test_mse', 'test_msle', 'test_r2', 'test_kappa']]
        print_metrics_los(los)
    else:
        print('& & & & & \\\\')

    return

if __name__=='__main__':
    los = 'ns_gat_los'
    ihm = 'ns_gat_ihm'
    main(los, ihm)