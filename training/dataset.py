import random
import os

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import BallTree

from utils import *

def add_time_decay(cfg, train_pos, graph_path):
    time_decay = cfg.model.get("time_decay", 0.0)
    time_min_weight = cfg.model.get("time_min_weight", 0.0)

    if time_decay == 0:
        weights = np.ones(len(train_pos))

    else:
        t_values = train_pos["time"].values.astype(np.float32)
        t_max = t_values.max()
        time_diff_days = (t_max - t_values) / (86400.0 * 1000)

        weights = (1 - time_min_weight) * np.exp(-time_decay * time_diff_days) + time_min_weight

        graph_path += f"_time_decay_{time_decay}_{time_min_weight}"

    return weights, graph_path

def add_geo_distance(cfg, train, n_items, graph_path):
    EARTH_RADIUS_KM = 6371.0
    geo_threshold = cfg.model.get("geo_threshold", 0.0)
    geo_sigma = cfg.model.get("geo_sigma", 0.0)
    geo_k_neighbors = cfg.model.get("geo_k_neighbors", 0)
    geo_weight = cfg.model.get("geo_weight", 1.0)

    if geo_threshold != 0 or geo_k_neighbors != 0:
        item_locs = train.drop_duplicates(subset=["iid"])[["iid", "latitude", "longitude"]].set_index("iid")
        item_locs = item_locs.reindex(range(n_items)).fillna(0)

        coords = np.radians(item_locs[["latitude", "longitude"]].values)
        tree = BallTree(coords, metric="haversine")

        if geo_threshold != 0:
            radius_rad = geo_threshold / EARTH_RADIUS_KM
            indices, dists_rad = tree.query_radius(coords, r=radius_rad, return_distance=True)
            graph_path += f"_geo_distance_threshold_{geo_threshold}"
        else:
            dists_rad, indices = tree.query(coords, k=geo_k_neighbors + 1)
            graph_path += f"_geo_distance_neighbors_{geo_k_neighbors}"

        if geo_sigma != 0.0:
            graph_path += f"_sigma_{geo_sigma}"
        elif geo_weight != 1.0:
            graph_path += f"_weight_{geo_weight}"

        rows, cols, weights = [], [], []

        for i, (neighbors, dists) in enumerate(zip(indices, dists_rad)):
            mask = neighbors != i
            valid_neighbors = neighbors[mask]
            valid_dists_rad = dists[mask]

            if len(valid_neighbors) == 0:
                continue

            if geo_sigma != 0.0:
                gamma = -1.0 / (geo_sigma ** 2)
                dists_km = valid_dists_rad * EARTH_RADIUS_KM
                w = np.exp(gamma * (dists_km ** 2))
            elif geo_weight != 1.0:
                w = np.ones(len(valid_neighbors), dtype=np.float32) * geo_weight
            else:
                w = np.ones(len(valid_neighbors), dtype=np.float32)

            rows.extend([i] * len(valid_neighbors))
            cols.extend(valid_neighbors)
            weights.extend(w)

        bottom_right = sp.coo_matrix((weights, (rows, cols)), shape=(n_items, n_items))
        bottom_right = bottom_right.tocsr()
        bottom_right = bottom_right.maximum(bottom_right.transpose())

    else:
        bottom_right = sp.csr_matrix((n_items, n_items))

    return bottom_right, graph_path

def build_adjacency_matrix(cfg):
    n_users, n_items = load_dataset_sizes(cfg)

    train = pd.read_csv(f"{cfg.paths.splits}/train.csv")
    train_pos = train[train["rating"] >= cfg.dataset.min_rating]
    u_ids = train_pos["uid"].values
    i_ids = train_pos["iid"].values

    graph_path = f"{cfg.paths.graph}/graph"
    weights, graph_path = add_time_decay(cfg, train_pos, graph_path)
    R = sp.coo_matrix((weights, (u_ids, i_ids)), shape=(n_users, n_items))

    bottom_right, graph_path = add_geo_distance(cfg, train, n_items, graph_path)

    top_left = sp.csr_matrix((n_users, n_users))
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
    torch.save(graph, f"{graph_path}.pt")

def load_adjacency_matrix(cfg):
    os.makedirs(cfg.paths.graph, exist_ok=True)
    graph_path = f"{cfg.paths.graph}/graph"

    time_decay = cfg.model.get("time_decay", 0.0)
    time_min_weight = cfg.model.get("time_min_weight", 0.0)

    if time_decay != 0:
        graph_path += f"_time_decay_{time_decay}_{time_min_weight}"

    geo_threshold = cfg.model.get("geo_threshold", 0.0)
    geo_sigma = cfg.model.get("geo_sigma", 0.0)
    geo_weight = cfg.model.get("geo_weight", 1.0)

    if geo_threshold != 0:
        graph_path += f"_geo_distance_threshold_{geo_threshold}"

    geo_k_neighbors = cfg.model.get("geo_k_neighbors", 0)
    if geo_k_neighbors != 0:
        graph_path += f"_geo_distance_neighbors_{geo_k_neighbors}"

    if geo_sigma != 0.0:
        graph_path += f"_sigma_{geo_sigma}"
    elif geo_weight != 1.0:
        graph_path += f"_weight_{geo_weight}"

    if not os.path.exists(f"{graph_path}.pt"):
        build_adjacency_matrix(cfg)

    graph = torch.load(f"{graph_path}.pt")
    return graph

class CuisineDataset(Dataset):
    def __init__(self, cfg):
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

        neg_item = self.sample_negative(user)
        return user, pos_item, neg_item

    def sample_negative(self, user):
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in self.user_pos_dict.get(user, set()):
                return neg_item

def construct_datasets(cfg):
    batch_size = cfg.training.batch_size

    train = CuisineDataset(cfg)
    train = DataLoader(train, batch_size=batch_size, shuffle=True)

    valid = pd.read_csv(f"{cfg.paths.splits}/valid.csv")
    test = pd.read_csv(f"{cfg.paths.splits}/test.csv")

    return {"train": train, "valid": valid, "test": test}