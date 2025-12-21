import os

import pandas as pd
import tqdm
import numpy as np

from utils import *

def apply_mappings(cfg, df):
    mappings = load_mappings(cfg, df)
    gmap2index, user2index = mappings["gmap2index"], mappings["user2index"]

    df["iid"] = df["gmap_id"].apply(lambda x: gmap2index[x])
    df["uid"] = df["user_id"].apply(lambda x: user2index[x])

    gmap2loc = get_gmap_to_loc(cfg)

    df["latitude"] = df["gmap_id"].apply(lambda x: gmap2loc[x][0])
    df["longitude"] = df["gmap_id"].apply(lambda x: gmap2loc[x][1])

    df = df[["uid", "iid", "time", "rating", "latitude", "longitude"]]

    return df

def get_gmap_to_loc(cfg):
    meta = pd.read_csv(f"{cfg.paths.combined}/meta.csv")

    gmap2loc = {gmap_id: (latitude, longitude) for (gmap_id, latitude, longitude) in meta[["gmap_id", "latitude", "longitude"]].values}

    return gmap2loc

def split_dataset_by_temporal(cfg):
    if os.path.exists(cfg.paths.splits):
        return

    df = pd.read_csv(f"{cfg.paths.combined}/review.csv")
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

    columns = ["uid", "iid", "rating", "time", "latitude", "longitude"]
    train_df = df.loc[train_indices][columns].reset_index(drop=True)
    valid_df = df.loc[valid_indices][columns].reset_index(drop=True)
    test_df = df.loc[test_indices][columns].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    os.makedirs(cfg.paths.splits, exist_ok=True)
    train_df.to_csv(f"{cfg.paths.splits}/train.csv", index=False)
    valid_df.to_csv(f"{cfg.paths.splits}/valid.csv", index=False)
    test_df.to_csv(f"{cfg.paths.splits}/test.csv", index=False)