import os
import gzip
import json
import random

import numpy as np
import pandas as pd
import torch

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def count_lines(path):
    with gzip.open(path, "r") as g:
        return sum(1 for _ in g)

def load_states(dataset_path):
    with open(f"{dataset_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    states = [state.replace(" ", "_") for state in states]
    return states


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def check_process_complete(cfg):
    if not os.path.exists(cfg.paths.review) or not os.path.exists(cfg.paths.meta):
        return False


    states = cfg.dataset.get("states", load_states(cfg.paths.dataset))
    num_states = len(states)
    num_review_files = len(os.listdir(cfg.paths.review))
    num_meta_files = len(os.listdir(cfg.paths.meta))

    return (num_states == num_review_files) and (num_states == num_meta_files)

def check_training_started(cfg):
    return os.path.exists(f"{cfg.paths.logs}/{cfg.name}.json")


def load_hydra_config(config_name):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="./config"):
        cfg = compose(config_name=config_name)

    return cfg


def save_mappings(cfg):
    df = load_combined_datast(cfg, "review")

    gmap_ids = np.unique(np.sort(df["gmap_id"].values))
    gmap2index = {gmap_id:index for index, gmap_id in enumerate(gmap_ids)}
    index2gmap = {index:gmap_id for index, gmap_id in enumerate(gmap_ids)}

    user_ids = np.unique(np.sort(df["user_id"].values))
    user2index = {user_id:index for index, user_id in enumerate(user_ids)}
    index2user = {index:user_id for index, user_id in enumerate(user_ids)}

    with open(f"{cfg.paths.combined}/mappings.json", "w") as f:
        json.dump({
            "gmap2index": gmap2index, "index2gmap": index2gmap,
            "user2index": user2index, "index2user": index2user
        }, f)

def load_mappings(cfg):
    if not os.path.exists(f"{cfg.paths.combined}/mappings.json"):
        save_mappings(cfg)

    with open(f"{cfg.paths.combined}/mappings.json", "r") as f:
        mappings = json.load(f)

    return mappings


def save_dataset_sizes(cfg):
    df = load_combined_datast(cfg, "review")
    gmap_ids = np.unique(np.sort(df["gmap_id"].values))
    user_ids = np.unique(np.sort(df["user_id"].values))

    n_items = gmap_ids.shape[0]
    n_users = user_ids.shape[0]

    os.makedirs(cfg.paths.result, exist_ok=True)
    with open(f"{cfg.paths.result}/dataset_size.txt", "w") as f:
        f.write(f"{n_users},{n_items}")

def load_dataset_sizes(cfg):
    if not os.path.exists(f"{cfg.paths.result}/dataset_size.txt"):
        save_dataset_sizes(cfg)

    with open(f"{cfg.paths.result}/dataset_size.txt", "r") as f:
        sizes = [int(l) for l in f.read().split(",")]

    return sizes


def load_splits_dataset(cfg):
    train_df = load_split_dataset(cfg, "train")
    train_df["split"] = train_df["uid"].apply(lambda x: "train")

    valid_df = load_split_dataset(cfg, "valid")
    valid_df["split"] = valid_df["uid"].apply(lambda x: "valid")

    test_df = load_split_dataset(cfg, "test")
    test_df["split"] = test_df["uid"].apply(lambda x: "test")

    df = pd.concat([train_df, valid_df, test_df], axis=0)

    return df

def load_split_dataset(cfg, split):
    test_df = pd.read_csv(f"{cfg.paths.splits}/{split}.csv")
    return test_df

def load_combined_datast(cfg, mode):
    df = pd.read_csv(f"{cfg.paths.combined}/{mode}.csv")
    return df