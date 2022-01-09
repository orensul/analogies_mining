import torch, argparse
# import os, json, tarfile, uuid, shutil
from collections import OrderedDict

# Modified from https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3

def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)

    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.

    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"

        return old_key

    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)

def rename_layernorm(param_name):
    new_name = param_name.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
    if new_name != param_name:
        print("Renamed '{}' to '{}'".format(param_name, new_name))
    return new_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the answer-first pipeline")
    parser.add_argument('--src', type=str, help = "Path to input params file.")
    parser.add_argument('--tgt', type=str, help = "Path to output params file.")
    args = parser.parse_args()
    rename_state_dict_keys(args.src, rename_layernorm, args.tgt)
