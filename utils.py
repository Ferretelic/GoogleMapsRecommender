import gzip
import json

class Config():
    def __init__(self, proj_path):
        self.min_rating = 4
        self.country = "Japanese"

        self.raw_path = f"{proj_path}/dataset/raw"
        self.processed_path = f"{proj_path}/dataset/processed"
        self.result_path = f"{proj_path}/results"

        self.meta_keys = [
            "gmap_id", "name", "address",
            "latitude", "longitude", "category",
            "avg_rating", "num_of_reviews", "price", "hours"
        ]

        self.review_keys = ["gmap_id", "user_id", "name", "time", "rating"]

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def load_states(config):
    with open(f"{config.raw_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    states = [state.replace(" ", "_") for state in states]
    return states