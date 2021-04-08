import numpy as np


def calc_delta_tx(obse, pred, percent):
    num_x = int(np.ceil(len(obse) * percent))
    return 2 * sorted(abs(np.array(obse) - np.array(pred)))[num_x - 1]


def calc_r_square(list_obse, list_pred):
    obse = np.array(list_obse)
    pred = np.array(list_pred)
    sse = np.sum(np.square(pred - obse))
    sst = np.sum(np.square(pred - np.mean(obse)))
    r_square = 1 - sse / sst
    return r_square
