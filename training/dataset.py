import sys
import random
import os

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

def save_dataset_sizes(cfg):
    df = pd.read_csv(f"{cfg.paths.country}/review.csv")
    gmap_ids = np.unique(np.sort(df["gmap_id"].values))
    user_ids = np.unique(np.sort(df["user_id"].values))

    n_items = gmap_ids.shape[0]
    n_users = user_ids.shape[0]

    os.makedirs(cfg.paths.result, exist_ok=True)
    with open(f"{cfg.paths.result}/dataset_size.txt", "w") as f:
        f.write(f"{n_users},{n_items}")

def load_dataset_sizes(cfg):
    if not os.path.exists(f"{cfg.paths.result}/dataset_size.txt"):
        save_dataset_sizes(cfg)

    with open(f"{cfg.paths.result}/dataset_size.txt", "r") as f:
        sizes = [int(l) for l in f.read().split(",")]

    return sizes

def build_adjacency_matrix(cfg):
    n_users, n_items = load_dataset_sizes(cfg)

    train = pd.read_csv(f"{cfg.paths.splits}/train.csv")
    train_pos = train[train["rating"] >= cfg.dataset.min_rating]
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

    torch.save(graph, f"{cfg.paths.result}/graph.pt")

def load_adjacency_matrix(cfg):
    if not os.path.exists(f"{cfg.paths.result}/graph.pt"):
        build_adjacency_matrix(cfg)

    graph = torch.load(f"{cfg.paths.result}/graph.pt")
    return graph

class CuisineDataset(Dataset):
    def __init__(self, cfg, hard_neg_prob):
        self.hard_neg_prob = hard_neg_prob

        _, self.n_items = load_dataset_sizes(cfg)

        df = pd.read_csv(f"{cfg.paths.splits}/train.csv")
        df_pos = df[df["rating"] >= cfg.dataset.min_rating]
        df_neg = df[df["rating"] < cfg.dataset.min_rating]

        self.user_pos_dict = df_pos.groupby("uid")["iid"].apply(set).to_dict()
        self.user_neg_dict = df_neg.groupby("uid")["iid"].apply(set).to_dict()

        self.interactions = df_pos[["uid", "iid"]].values
        self.df_pos = df_pos

    def __len__(self):
        return self.interactions.shape[0]

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]

        neg_item = self._sample_negative(user)
        return user, pos_item, neg_item

    def _sample_negative(self, user):
        explicit_negs = self.user_neg_dict.get(user, set())

        if len(explicit_negs) > 0 and random.random() < self.hard_neg_prob:
            return np.random.choice(list(explicit_negs))

        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in self.user_pos_dict.get(user, set()):
                return neg_item

def construct_datasets(cfg):
    batch_size = cfg.training.batch_size

    train = CuisineDataset(cfg, hard_neg_prob=cfg.training.hard_neg_prob)
    train = DataLoader(train, batch_size=batch_size, shuffle=True)

    valid = pd.read_csv(f"{cfg.paths.splits}/valid.csv")
    test = pd.read_csv(f"{cfg.paths.splits}/test.csv")

    return {"train": train, "valid": valid, "test": test}