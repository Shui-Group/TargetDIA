import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pdeep2.load_param import load_pdeep2_param
from models.pdeep2.train import train_pdeep2
from models.pdeep2.predict import predict_pdeep2


if __name__ == '__main__':

    if len(sys.argv) != 1:
        param_path = sys.argv[1]
        task = sys.argv[2]  # This can be training / eval / predict
    else:
        param_path = r''

        task = 'predict'  # This can be training / eval / predict

    if os.path.isfile(param_path):
        # Load config from given json file
        param = load_pdeep2_param(param_path)
    else:
        raise FileNotFoundError(f'The config path argument - {param_path} - is not a valid file')

    if task == 'training':
        train_pdeep2(param=param)
    elif task == 'eval':
        pass
    elif task == 'predict':
        predict_pdeep2(param=param)
    else:
        raise ValueError('The second argument should be defined as a task name: training / eval / predict')
