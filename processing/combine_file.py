import os

import pandas as pd

from utils import *

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
    df.to_csv(file_path, index=False)