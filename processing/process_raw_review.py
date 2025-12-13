import os
from functools import partial

import numpy as np
import pandas as pd

from utils import *

def filter_by_gmap_id(data, gmap_ids):
    gmap_id = data.get("gmap_id", None)
    if gmap_id is None:
        return False

    return gmap_id in gmap_ids

def filter_raw_state_review_data(state, filters):
    raw_file = f"{raw_path}/review/{state}.json.gz"
    if not os.path.exists(raw_file):
        return

    os.makedirs(f"{processed_path}/{country}/review/", exist_ok=True)
    processed_file = f"{processed_path}/{country}/review/{state}.csv"
    if os.path.exists(processed_file):
        return

    print(f"Start filtering {state}")
    reviews = []
    for review in parse(raw_file):
        if all([f(data=review) for f in filters]):
            review = {key: review.get(key, None) for key in review_keys}
            reviews.append(review)

    print(f"We obtained total of {len(reviews)} reviews after filtering by gmap_id")
    df = pd.DataFrame(reviews)
    df.to_csv(processed_file, index=False)
    return df

def filter_raw_review_data():
    states = load_states()

    for state in states:
        gmap_ids = set(pd.read_csv(f"{processed_path}/{country}/meta/{state}.csv")["gmap_id"].values)
        gmap_id_filter = partial(filter_by_gmap_id, gmap_ids=gmap_ids)
        filter_raw_state_review_data(state, [gmap_id_filter])

if __name__ == "__main__":
    filter_raw_review_data()