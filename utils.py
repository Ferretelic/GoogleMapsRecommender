import os
import gzip
import json

import numpy as np

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def load_states(raw_path):
    with open(f"{raw_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    states = [state.replace(" ", "_") for state in states]
    return states

def check_process_complete(cfg):
    num_states = len(load_states(cfg.paths.raw))
    num_review_files = len(os.listdir(cfg.paths.review))
    num_meta_files = len(os.listdir(cfg.paths.meta))

    return (num_states == num_review_files) and (num_states == num_meta_files)

def check_training_started(cfg):
    return os.path.exists(f"{cfg.paths.logs}/{cfg.name}.json")

def check_training_completed(cfg):
    with open(f"{cfg.paths.logs}/{cfg.name}.json", "r") as f:
        logs = json.load(f)

    early_stopping = cfg.training.early_stopping
    ndcgs = np.array(logs["valid"]["ndcg"])
    n_patience = ndcgs.shape[0] - (np.argmax(ndcgs) + 1)

    if early_stopping == n_patience:
        print("        Training has been already completed.")
        return True

    else:
        print(f"        Training is still in progress with patience {n_patience}.")
