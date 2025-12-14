import sys
import os

import pandas as pd
import tqdm

sys.path.append("..")
from utils import *

def positive_split(df):
    df = df[df["rating"] >= min_rating]
    print(f"Finished extracting {df.shape[0]} positive interactions")
    return df

def temporal_split(df):
    df = df.sort_values(by=["user_id", "time"])
    groups = df.groupby("user_id")

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

    train_df = df.loc[train_indices].reset_index(drop=True)
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    os.makedirs(f"{processed_path}/{country}/splits", exist_ok=True)
    train_df.to_csv(f"{processed_path}/{country}/splits/train.csv", index=False)
    valid_df.to_csv(f"{processed_path}/{country}/splits/valid.csv", index=False)
    test_df.to_csv(f"{processed_path}/{country}/splits/test.csv", index=False)

if __name__ == "__main__":
    df = pd.read_csv(f"{processed_path}/{country}/review.csv")
    df = positive_split(df)
    temporal_split(df)