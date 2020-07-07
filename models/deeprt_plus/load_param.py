import json


def load_deeprt_param(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f, )
    return config
