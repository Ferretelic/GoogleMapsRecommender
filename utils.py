import os
import gzip
import json

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
