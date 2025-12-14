import os

import pandas as pd

from utils import *

def filter_reviews_by_users(cfg, df):
    user_counts = df.groupby("user_id")["gmap_id"].count()

    valid_users = user_counts[user_counts >= cfg.dataset.min_num_reviews].index
    df = df[df["user_id"].isin(valid_users)].reset_index(drop=False)

    return df

def combine_state_files(cfg, mode):
    file_path = f"{cfg.paths.country}/{mode}.csv"

    if os.path.exists(file_path):
        return

    states = load_states(cfg.paths.raw)

    all_data = []
    for state in states:
        data = pd.read_csv(f"{cfg.paths.country}/{mode}/{state}.csv")
        data["state"] = data["gmap_id"].apply(lambda x: state)
        all_data.append(data)

    df = pd.concat(all_data, axis=0)

    if mode == "review":
        df = filter_reviews_by_users(cfg, df)

    df.to_csv(file_path, index=False)