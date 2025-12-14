import sys

import pandas as pd

sys.path.append("..")
from utils import *

def combine_state_files(config, mode):
    states = load_states(config)

    all_data = []
    for state in states:
        data = pd.read_csv(f"{config.processed_path}/{config.country}/{mode}/{state}.csv")
        data["state"] = data["gmap_id"].apply(lambda x: state)
        all_data.append(data)

    df = pd.concat(all_data, axis=0)
    df.to_csv(f"{config.processed_path}/{config.country}/{mode}.csv", index=False)

if __name__ == "__main__":
    config = Config("..")
    combine_state_files(config, "meta")
    combine_state_files(config, "review")