import json


def load_pdeep2_param(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f, )
    return config
