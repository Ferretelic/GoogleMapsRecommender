import os
from functools import partial

import pandas as pd

from utils import *

def filter_by_country(country, data):
    category = data.get("category", None)
    if category is None:
        return False

    return any([country.lower() in c.lower().split() for c in category])

def filter_by_num_reviews(data, min_num_reviews):
    return data["num_of_reviews"] >= min_num_reviews

def filter_raw_state_meta_data(cfg, state, filters):
    raw_file = f"{cfg.paths.raw}/meta/{state}.json.gz"
    if not os.path.exists(raw_file):
        return

    meta_path = f"{cfg.paths.processed}/{cfg.dataset.country}/meta/"
    os.makedirs(meta_path, exist_ok=True)
    processed_file = f"{meta_path}/{state}.csv"
    if os.path.exists(processed_file):
        return

    print(f"Start filtering {state}")

    meta_list = []
    for meta in parse(raw_file):
        if all([f(data=meta) for f in filters]):
            meta = {key: meta.get(key, None) for key in cfg.dataset.meta_keys}
            meta_list.append(meta)

    print(f"We obtained total of {len(meta_list)} {cfg.dataset.country} restaurants after filtering")

    df = pd.DataFrame(meta_list)
    df.to_csv(processed_file, index=False)

def filter_raw_meta_data(cfg, filters):
    states = load_states(cfg.paths.raw)

    for state in states:
        filter_raw_state_meta_data(cfg, state, filters)