import os
import sys

sys.path.append('../')

from models.deeprt_plus import load_param
from models.deeprt_plus import train_deeprt_plus
from models.deeprt_plus import pred_deeprt_plus

if __name__ == '__main__':

    if len(sys.argv) != 1:
        param_path = sys.argv[1]
        task = sys.argv[2]  # This can be training / eval / predict
    else:
        param_path = r''
        task = 'predict'  # This can be training / eval / predict

    if os.path.isfile(param_path):
        # Load config from given json file
        param = load_param.load_deeprt_param(param_path)
    else:
        raise FileNotFoundError(f'The config path argument - {param_path} - is not a valid file')

    if task == 'training':
        train_deeprt_plus.train_deeprt(param)
    elif task == 'eval':
        pass
    elif task == 'predict':
        pred_deeprt_plus.deeprt_pred(param)
    else:
        raise ValueError('The second argument should be defined as a task name: training / eval / predict')
