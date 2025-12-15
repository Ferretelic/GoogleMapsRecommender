import os
import gzip
import json
import math

import numpy as np
import torch

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def load_states(raw_path):
    with open(f"{raw_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    states = [state.replace(" ", "_") for state in states]
    return states

def check_process_complete(cfg):
    num_states = len(load_states(cfg.paths.raw))
    num_review_files = len(os.listdir(cfg.paths.review))
    num_meta_files = len(os.listdir(cfg.paths.meta))

    return (num_states == num_review_files) and (num_states == num_meta_files)

def check_training_started(cfg):
    return os.path.exists(f"{cfg.paths.logs}/{cfg.name}.json")

def check_training_completed(cfg):
    with open(f"{cfg.paths.logs}/{cfg.name}.json", "r") as f:
        logs = json.load(f)

    early_stopping = cfg.training.early_stopping
    ndcgs = np.array(logs["valid"]["ndcg"])
    n_patience = ndcgs.shape[0] - (np.argmax(ndcgs) + 1)

    if early_stopping == n_patience:
        print("    Training has been already completed.")
        return True

    else:
        print(f"    Training is still in progress with patience {n_patience}.")

def split_embeddings(cfg):
    model_path = f"{cfg.paths.embedding}/{cfg.name}"
    file_size = os.path.getsize(f"{model_path}.pt") / (1024 ** 2)

    if file_size <= 100:
        return

    target_chunk_size_mb = 50
    num_chunks = math.ceil(file_size / target_chunk_size_mb)

    if os.path.exists(f"{cfg.paths.embedding}/{cfg.name}"):
        return

    print(f"    Splitting into {num_chunks} chunks")
    users_emb, items_emb = torch.load(f"{model_path}.pt")
    users_chunks = torch.chunk(users_emb, num_chunks, dim=1)

    os.makedirs(model_path, exist_ok=True)
    torch.save(items_emb.clone(), f"{model_path}/items.pt")
    for i in range(num_chunks):
        torch.save(users_chunks[i].clone(), f"{model_path}/users_{i}.pt")

def load_embeddings(cfg):
    model_path = f"{cfg.paths.embedding}/{cfg.name}"

    if os.path.exists(model_path):
        i = 0
        users_chunks = []
        while True:
            user_path = f"{model_path}/users_{i}.pt"

            if not os.path.exists(user_path):
                break

            users_chunks.append(torch.load(user_path))
            i += 1

        users_emb = torch.cat(users_chunks, dim=1)
        items_emb = torch.load(f"{model_path}/items.pt")

    else:
        users_emb, items_emb = torch.load(model_path)

    return users_emb, items_emb