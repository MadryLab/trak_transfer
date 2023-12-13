import yaml
import numpy as np

def read_yaml(f):
    with open(f, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def _get_inds(split, c_args):
    key = f"{split}_inds_path"
    if key in c_args:
        return np.load(c_args[key])
    else:
        return None