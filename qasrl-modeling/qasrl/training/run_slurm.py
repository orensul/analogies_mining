#!/usr/bin/env python3

import importlib
from allennlp.common.util import import_submodules
importlib.invalidate_caches()
import sys
sys.path.append(".")
import_submodules("qasrl")

import subprocess
import os
import argparse
import torch
from allennlp.common import Params
from allennlp.commands.train import train_model

def main():
    parser = argparse.ArgumentParser(description = "Train QA-SRL model variants.")
    parser.add_argument("models_root", metavar = "path", type = str, help = "Path to root of model variants")
    parser.add_argument("models_branch", metavar = "path", type = str, help = "Path to config file")
    parser.add_argument("initial_batch_size", metavar = "n", type = int, help = "Batch size to start with before cutting as necessary")
    args = parser.parse_args()

    serialization_directory = "/gscratch/cse/julianjm/qasrl-models/" + args.models_branch
    current_batch_size = args.initial_batch_size
    done = False
    while not done:
        try:
            if not os.path.exists(serialization_directory + "/current"):
                print("Starting new training round", flush = True)
                if not os.path.exists(serialization_directory):
                    os.makedirs(serialization_directory)
                config_path = args.models_root + "/" + args.models_branch
                params = Params.from_file(config_path, "")
                params["trainer"]["num_serialized_models_to_keep"] = 2
                params["trainer"]["should_log_parameter_statistics"] = False
                params["iterator"]["biggest_batch_first"] = True
                params["iterator"]["batch_size"] = current_batch_size
                torch.cuda.empty_cache()
                train_model(params,
                            serialization_directory + "/current",
                            file_friendly_logging = True)
                done = True
            else:
                print("Recovering from a previously preempted run", flush = True)
                config_path = serialization_directory + "/current/config.json"
                params = Params.from_file(config_path, "")
                current_batch_size = params["iterator"]["batch_size"]
                torch.cuda.empty_cache()
                train_model(params,
                            serialization_directory + "/current",
                            file_friendly_logging = True,
                            recover = True)
                done = True
        except RuntimeError as e:

            if 'out of memory' in str(e) or "CUDNN_STATUS_NOT_SUPPORTED" in str(e) or "an illegal memory access was encountered" in str(e):
                print(str(e), flush = True)
                print('Reducing batch size to %s and retrying' % (current_batch_size / 2), flush = True)
                subprocess.call(["mv", serialization_directory + "/current", serialization_directory + "/" + str(int(current_batch_size))])
                current_batch_size = current_batch_size / 2
                torch.cuda.empty_cache()
            else:
                raise e
    print("Finished training.", flush = True)

if __name__ == "__main__":
    main()
