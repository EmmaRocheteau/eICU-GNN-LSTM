from scipy import stats
from src.significance_testing.load_and_inspect import get_results

def summary(file1, file2):
    df1, metrics = get_results(file1)
    df2, metrics = get_results(file2)
    for metric in metrics:
        sig_ttest(df1[metric], df2[metric], metric, file1, file2)

def sig_ttest(list_results1, list_results2, metric, file1, file2):
    t2, p2 = stats.ttest_ind(list_results1, list_results2)
    #print("p = " + str(p2))
    stars = ''
    better = ''
    if p2 < 0.05:
        stars +='*'
        if p2 < 0.001:
            stars += '*'
        if list_results1.mean() > list_results2.mean():
            better = file1[:-4]
        else:
            better = file2[:-4]
    print('{}: {} {} is higher'.format(metric, stars, better))
    return

if __name__=='__main__':
    file1 = 'lstm_los_no_diag.csv'
    file2 = 'lstm_los.csv'
    summary(file1, file2)