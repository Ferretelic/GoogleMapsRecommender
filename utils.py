import gzip
import json

min_rating = 4
country = "Japanese"

raw_path = "../dataset/raw"
processed_path = "../dataset/processed"

meta_keys = ["gmap_id", "name", "address", "latitude", "longitude", "category", "avg_rating", "num_of_reviews", "price", "hours"]
review_keys = ["gmap_id", "user_id", "name", "time", "rating"]

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def load_states():
    with open(f"{raw_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    states = [state.replace(" ", "_") for state in states]
    return states