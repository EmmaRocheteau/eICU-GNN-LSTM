import pandas as pd

results_dir = 'results/'

def get_results(file):

    df = pd.read_csv(results_dir + file)
    metrics = [a for a in list(df) if ('test_'in a and not 'lstm' in a and not 'tot' in a and not 'all' in a)]
    return df, metrics

if __name__=='__main__':
    file = 'lstm_los_no_diag.csv'
    df1, metrics = get_results(file)
    print('mean')
    print(df1[metrics].mean())
    print('sd')
    print(df1[metrics].std())
    print('max')
    print(df1[metrics].max())
    print('min')
    print(df1[metrics].min())