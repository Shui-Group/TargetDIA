from math import sqrt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def RMSE(act, pred):
    '''
    accept two numpy arrays
    '''
    return sqrt(np.mean(np.square(act - pred)))


def Pearson(act, pred):
    return pearsonr(act, pred)[0]


def Spearman(act, pred):
    '''
    Note: there is no need to use spearman correlation for now
    '''
    return spearmanr(act, pred)[0]


def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]


def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))
