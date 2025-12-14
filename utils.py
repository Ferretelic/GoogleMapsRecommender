import gzip
import json

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def load_states(raw_path):
    with open(f"{raw_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    states = [state.replace(" ", "_") for state in states]
    return states