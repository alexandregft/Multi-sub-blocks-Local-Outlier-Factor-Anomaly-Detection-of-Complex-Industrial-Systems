import os
import yaml


def read_config(file_path):
    with open(os.path.normpath(file_path), 'r') as f:
        opts = yaml.load(f, Loader=yaml.FullLoader)

    return opts