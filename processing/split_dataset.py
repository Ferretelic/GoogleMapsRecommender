import os

import pandas as pd
import tqdm
import numpy as np

from utils import *

def save_mappings(cfg, df):
    gmap_ids = np.unique(np.sort(df["gmap_id"].values))
    gmap2index = {gmap_id:index for index, gmap_id in enumerate(gmap_ids)}
    index2gmap = {index:gmap_id for index, gmap_id in enumerate(gmap_ids)}

    user_ids = np.unique(np.sort(df["user_id"].values))
    user2index = {user_id:index for index, user_id in enumerate(user_ids)}
    index2user = {index:user_id for index, user_id in enumerate(user_ids)}

    with open(f"{cfg.paths.country}/mappings.json", "w") as f:
        json.dump({
            "gmap2index": gmap2index, "index2gmap": index2gmap,
            "user2index": user2index, "index2user": index2user
        }, f)

def load_mappings(cfg, df):
    if not os.path.exists(f"{cfg.paths.country}/mappings.json"):
        save_mappings(cfg, df)

    with open(f"{cfg.paths.country}/mappings.json", "r") as f:
        mappings = json.load(f)

    return mappings["gmap2index"], mappings["user2index"]

def apply_mappings(cfg, df):
    gmap2index, user2index = load_mappings(cfg, df)

    df["iid"] = df["gmap_id"].apply(lambda x: gmap2index[x])
    df["uid"] = df["user_id"].apply(lambda x: user2index[x])

    df = df[["uid", "iid", "time", "rating"]]

    return df

def split_dataset_by_temporal(cfg):
    if os.path.exists(cfg.paths.splits):
        return

    df = pd.read_csv(f"{cfg.paths.country}/review.csv")
    df = df[df["rating"] >= cfg.dataset.min_rating]
    df = apply_mappings(cfg, df)

    df = df.sort_values(by=["uid", "time"])
    groups = df.groupby("uid")

    train_indices = []
    valid_indices = []
    test_indices = []

    for _, group in tqdm.tqdm(groups):
        indices = group.index.tolist()

        if len(indices) < 3:
            train_indices.extend(indices)
            continue

        test_indices.append(indices[-1])
        valid_indices.append(indices[-2])
        train_indices.extend(indices[:-2])

    train_df = df.loc[train_indices][["uid", "iid", "rating"]].reset_index(drop=True)
    valid_df = df.loc[valid_indices][["uid", "iid", "rating"]].reset_index(drop=True)
    test_df = df.loc[test_indices][["uid", "iid", "rating"]].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    os.makedirs(cfg.paths.splits, exist_ok=True)
    train_df.to_csv(f"{cfg.paths.splits}/train.csv", index=False)
    valid_df.to_csv(f"{cfg.paths.splits}/valid.csv", index=False)
    test_df.to_csv(f"{cfg.paths.splits}/test.csv", index=False)