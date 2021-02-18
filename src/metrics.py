import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.metrics import classification, regression
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred), torch.log(actual))


def get_loss_function(task, class_weights):
    if task == 'ihm':
        if class_weights is False:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return MSLELoss()


def define_metrics(task):
    m = {}
    if task == 'ihm':
        # these take labels:
        m['acc'] = classification.Accuracy()
        #m['auroc'] = classification.AUROC()
    else:
        m['mae'] = regression.MAE()
        m['rmse'] = regression.RMSE()
        m['rmlse'] = regression.RMSLE()
    return m


class CustomBins:
    inf = 1e18
    # 10 classes
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(4/24, y_true))) * 100  # this stops the mape being a stupidly large value when y_true happens to be very small


def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean(np.square(np.log(y_true/y_pred)))


def get_per_node_result(y_true, pred_prob, ids, is_cls):
    if is_cls:
        return per_node_class_result(y_true, pred_prob, ids)
    else:
        return per_node_reg_result(y_true, pred_prob, ids)


def per_node_class_result(y_true, pred_prob, ids):
    """collect per node results:
    for classification - bool"""
    if isinstance(ids, range):
        ids = np.array(ids)
    labels = pred_prob.argmax(axis=1)
    # 
    per_node = {}
    per_node['ids'] = ids.astype(int)
    per_node['true'] = y_true.astype(int)
    per_node['pred'] = labels.astype(int)
    per_node['correct_cls'] = (labels == y_true).astype(int)
    per_node['correct_0'] = per_node['correct_cls'] & (y_true == 0)
    per_node['correct_1'] = per_node['correct_cls'] & (y_true == 1)
    return per_node


def per_node_reg_result(y_true, pred, ids):
    """collect per node results:
    for regression - squared error, percentage error """
    # 
    if isinstance(ids, range):
        ids = np.array(ids)
    per_node = {}
    per_node['ids'] = ids.astype(int)
    per_node['y_true'] = y_true
    per_node['pred'] = pred
    per_node['err'] = (y_true - pred)
    per_node['abs_err'] = np.abs((y_true - pred))  # absolute error
    per_node['per_err'] = np.abs((y_true - pred) / np.maximum(4/24, y_true))
    per_node['sq_err'] = np.square((y_true - pred))  # square error
    per_node['sq_log_err'] = np.square(np.log(y_true/pred)) # square log error
    return per_node


def get_metrics(y_true, pred_prob, verbose, is_cls):
    if is_cls:
        return compute_binary_metrics(y_true, pred_prob, verbose)
    else:
        return print_metrics_regression(y_true, pred_prob, verbose)

def print_metrics_regression(y_true, predictions, verbose):
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = confusion_matrix(y_true_bins, prediction_bins, labels=range(10))
    if verbose:
        print('Custom bins confusion matrix:')
        print(cf)
    
    results = {}
    results['kappa'] = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    results['mad'] = metrics.mean_absolute_error(y_true, predictions)
    results['mse'] = metrics.mean_squared_error(y_true, predictions)
    results['mape'] = mean_absolute_percentage_error(y_true, predictions)
    results['msle'] = mean_squared_logarithmic_error(y_true, predictions)
    results['r2'] = metrics.r2_score(y_true, predictions)
    results = {key: float(results[key]) for key in results}
    if verbose:
        for key in results:
            print("{}: {:.4f}".format(key, results[key]))
    return results
    

def compute_binary_metrics(y, pred_prob, verbose):
    pred_prob = np.array(pred_prob)
    labels = pred_prob.argmax(axis=1)
    cf = confusion_matrix(y, labels, labels=range(2))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)
    
    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    if len(set(y)) != 1:
        auroc = metrics.roc_auc_score(y, pred_prob[:, 1])
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y, pred_prob[:, 1])
        auprc = metrics.auc(recalls, precisions)
        minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
        f1macro = metrics.f1_score(y, labels, average='macro')
    else:
        auroc = np.nan
        auprc = np.nan
        minpse = np.nan
        f1macro = np.nan

    results = {"acc": acc, "prec0": prec0, "prec1": prec1, "rec0": rec0, "rec1": rec1,
               "auroc": auroc, "auprc": auprc, "minpse": minpse, 
               "f1macro": f1macro}
    results = {key: float(results[key]) for key in results}
    if verbose:
        for key in results:
            print("{}: {:.4f}".format(key, results[key]))

    return results
