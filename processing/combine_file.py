import sys

import pandas as pd

sys.path.append("..")
from utils import *

def combine_state_files(mode):
    states = load_states()

    all_data = []
    for state in states:
        data = pd.read_csv(f"{processed_path}/{country}/{mode}/{state}.csv")
        data["state"] = data["gmap_id"].apply(lambda x: state)
        all_data.append(data)

    df = pd.concat(all_data, axis=0)
    df.to_csv(f"{processed_path}/{country}/{mode}.csv", index=False)

if __name__ == "__main__":
    combine_state_files("meta")
    combine_state_files("review")