import sys
import os

import pandas as pd
import tqdm
import numpy as np

sys.path.append("..")
from utils import *

def apply_mappings(df):
    gmap_ids = np.unique(np.sort(df["gmap_id"].values))
    gmap2index = {gmap_id:index for index, gmap_id in enumerate(gmap_ids)}

    user_ids = np.unique(np.sort(df["user_id"].values))
    user2index = {user_id:index for index, user_id in enumerate(user_ids)}

    df["iid"] = df["gmap_id"].apply(lambda x: gmap2index[x])
    df["uid"] = df["user_id"].apply(lambda x: user2index[x])

    df = df[["uid", "iid", "time", "rating"]]

    return df

def temporal_split(config, df):
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

    folder_path = f"{config.processed_path}/{config.country}/splits/"
    os.makedirs(folder_path, exist_ok=True)
    train_df.to_csv(f"{folder_path}train.csv", index=False)
    valid_df.to_csv(f"{folder_path}/valid.csv", index=False)
    test_df.to_csv(f"{folder_path}/test.csv", index=False)

if __name__ == "__main__":
    config = Config("..")
    df = pd.read_csv(f"{config.processed_path}/{config.country}/review.csv")
    df = apply_mappings(df)
    temporal_split(config, df)