import sys
import os
from functools import partial

import pandas as pd

sys.path.append("..")
from utils import *

def filter_by_country(config, data):
    category = data.get("category", None)
    if category is None:
        return False

    return any([config.country.lower() in c.lower().split() for c in category])

def filter_by_num_reviews(data, min_num_reviews):
    return data["num_of_reviews"] >= min_num_reviews

def filter_raw_state_meta_data(config, state, filters):
    raw_file = f"{config.raw_path}/meta/{state}.json.gz"
    if not os.path.exists(raw_file):
        return

    os.makedirs(f"{config.processed_path}/{config.country}/meta/", exist_ok=True)
    processed_file = f"{config.processed_path}/{config.country}/meta/{state}.csv"
    if os.path.exists(processed_file):
        return

    print(f"Start filtering {state}")

    meta_list = []
    for meta in parse(raw_file):
        if all([f(data=meta) for f in filters]):
            meta = {key: meta.get(key, None) for key in config.meta_keys}
            meta_list.append(meta)

    print(f"We obtained total of {len(meta_list)} {config.country} restaurants after filtering")

    df = pd.DataFrame(meta_list)
    df.to_csv(processed_file, index=False)

def filter_raw_meta_data(config, filters):
    states = load_states(config)

    for state in states:
        filter_raw_state_meta_data(config, state, filters)

if __name__ == "__main__":
    config = Config("..")
    min_num_reviews = 5

    country_filter = partial(filter_by_country, config=config)
    num_reviews_filter = partial(filter_by_num_reviews, min_num_reviews=min_num_reviews)
    filter_raw_meta_data(config, [country_filter, num_reviews_filter])