import sys
import random
import os

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

sys.path.append("..")
from utils import *

def save_dataset_sizes(config):
    df = pd.read_csv(f"{config.processed_path}/{config.country}/review.csv")
    gmap_ids = np.unique(np.sort(df["gmap_id"].values))
    user_ids = np.unique(np.sort(df["user_id"].values))

    n_items = gmap_ids.shape[0]
    n_users = user_ids.shape[0]

    os.makedirs(config.result_path, exist_ok=True)
    with open(f"{config.result_path}/dataset_size.txt", "w") as f:
        f.write(f"{n_users},{n_items}")

def load_dataset_sizes(config):
    if not os.path.exists(f"{config.result_path}/dataset_size.txt"):
        save_dataset_sizes(config)

    with open(f"{config.result_path}/dataset_size.txt", "r") as f:
        sizes = [int(l) for l in f.read().split(",")]

    return sizes

def build_adjacency_matrix(config):
    n_users, n_items = load_dataset_sizes(config)

    train = pd.read_csv(f"{config.processed_path}/{config.country}/splits/train.csv")
    train_pos = train[train["rating"] >= config.min_rating]
    u_ids = train_pos["uid"].values
    i_ids = train_pos["iid"].values

    R = sp.coo_matrix((np.ones(len(u_ids)), (u_ids, i_ids)), shape=(n_users, n_items))

    top_left = sp.csr_matrix((n_users, n_users))
    bottom_right = sp.csr_matrix((n_items, n_items))

    adj_mat = sp.vstack([sp.hstack([top_left, R]), sp.hstack([R.T, bottom_right])])

    rowsum = np.array(adj_mat.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)

    mask = rowsum > 0
    d_inv_sqrt[mask] = np.power(rowsum[mask], -0.5)

    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)

    coo = norm_adj.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))

    graph = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()

    torch.save(graph, f"{config.result_path}/graph.pt")

def load_adjacency_matrix(config):
    if not os.path.exists(f"{config.result_path}/graph.pt"):
        build_adjacency_matrix(config)

    graph = torch.load(f"{config.result_path}/graph.pt")
    return graph

class CuisineDataset(Dataset):
    def __init__(self, config, mode, hard_neg_prob=None):
        self.mode = mode
        self.hard_neg_prob = hard_neg_prob

        _, self.n_items = load_dataset_sizes(config)

        df = pd.read_csv(f"{config.processed_path}/{config.country}/splits/{mode}.csv")

        df_pos = df[df["rating"] >= config.min_rating]
        self.interactions = df_pos[["uid", "iid"]].values

        if mode == "train":
            df_neg = df[df["rating"] < config.min_rating]

            self.user_pos_dict = df_pos.groupby("uid")["iid"].apply(set).to_dict()
            self.user_neg_dict = df_neg.groupby("uid")["iid"].apply(set).to_dict()

    def __len__(self):
        return self.interactions.shape[0]

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]

        if self.mode == "train":
            neg_item = self._sample_negative(user)
            return user, pos_item, neg_item
        else:
            return user, pos_item

    def _sample_negative(self, user):
        explicit_negs = self.user_neg_dict.get(user, set())

        if len(explicit_negs) > 0 and random.random() < self.hard_neg_prob:
            return np.random.choice(list(explicit_negs))

        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in self.user_pos_dict.get(user, set()):
                return neg_item

if __name__ == "__main__":
    config = Config("..")
    graph = load_adjacency_matrix(config)
    hard_neg_prob = 0.5
    dataset = CuisineDataset(config, "train", hard_neg_prob)

    print(f"Graph indices size: {graph.indices().shape}")
    u, p, n = dataset[0]
    print(f"Sample: User={u}, Pos={p}, Neg={n}")