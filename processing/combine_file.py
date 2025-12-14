import os

import pandas as pd

from utils import *

def filter_reviews_with_k_core(cfg, df):
    print("    Start filtering with k core")
    min_num_reviews = cfg.dataset.min_num_reviews
    df_core = df.copy()

    while True:
        start_shape = df_core.shape

        user_counts = df_core.groupby("user_id")["gmap_id"].count()
        valid_users = user_counts[user_counts >= min_num_reviews].index
        df_core = df_core[df_core["user_id"].isin(valid_users)]

        item_counts = df_core.groupby("gmap_id")["user_id"].count()
        valid_items = item_counts[item_counts >= min_num_reviews].index
        df_core = df_core[df_core["gmap_id"].isin(valid_items)]

        end_shape = df_core.shape
        print(f"        Interactions: before {start_shape[0]} vs after {end_shape[0]}")

        if start_shape == end_shape:
            break

    return df_core

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
        df = filter_reviews_with_k_core(cfg, df)

    df.to_csv(file_path, index=False)