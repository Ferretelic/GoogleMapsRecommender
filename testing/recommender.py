import json

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

class Recommender():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.test.device)

        train_df = pd.read_csv(f"{cfg.paths.splits}/train.csv", dtype={"uid": int, "iid": int})
        self.train_user_pos = train_df.groupby("uid")["iid"].apply(set).to_dict()

    def mask_train_items(self, scores, users):
        users = users.cpu().numpy()
        rows, cols = [], []
        for i, uid in enumerate(users):
            train_items = self.train_user_pos.get(uid, set())
            train_items = list(train_items)
            if train_items:
                rows.extend([i] * len(train_items))
                cols.extend(train_items)

        if rows:
            scores[rows, cols] = -float("inf")

        item_size = self.load_item_size()
        mapping_size = self.load_mapping_size()

        return scores

    def load_mapping_size(self):
        with open(f"{self.cfg.paths.combined}/mappings.json", "r") as f:
            mappings = json.load(f)

        return len(mappings["index2gmap"])

    def load_item_size(self):
        with open(f"{self.cfg.paths.result}/dataset_size.txt", "r") as f:
            _, item_size = [int(l) for l in f.read().split(",")]

        return item_size

    def get_top_k_items(self, scores, rank):
        _, topk_indices = torch.topk(scores, k=rank, dim=1)
        return topk_indices

    def recommend(self, users, rank):
        scores = self.compute_scores(users)
        scores = self.mask_train_items(scores, users)
        topk_indices = self.get_top_k_items(scores, rank)

        return topk_indices

class LocationRecommender(Recommender):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.item_popularity = self.load_item_popularity(cfg)
        self.item_location = self.load_item_location(cfg)


    def load_item_popularity(self, cfg):
        item_size = self.load_item_size()
        item_popularity = torch.zeros(item_size, dtype=torch.float32).to(self.device)

        train_df = pd.read_csv(f"{self.cfg.paths.splits}/train.csv")
        for iid, count in train_df["iid"].value_counts().reset_index().values:
            item_popularity[iid] = count

        return item_popularity

    def load_item_location(self, cfg):
        item_size = self.load_item_size()
        item_location = torch.zeros((item_size, 2), dtype=torch.float32)

        train_df = pd.read_csv(f"{self.cfg.paths.splits}/train.csv")
        valid_df = pd.read_csv(f"{self.cfg.paths.splits}/valid.csv")
        test_df = pd.read_csv(f"{self.cfg.paths.splits}/test.csv")
        df = pd.concat([train_df, valid_df, test_df], axis=0)
        df = df.groupby("iid")[["latitude", "longitude"]].first().reset_index()

        for (iid, latitude, longitude) in df.values:
            item_location[int(iid), :] = torch.tensor(np.radians((latitude, longitude)), dtype=torch.float32)

        item_location = item_location.to(self.device)

        return item_location

    def get_user_centroids(self, users):
        batch_centroids = torch.zeros(users.size(0), 2, device=self.device)

        for i, uid in enumerate(users.cpu().numpy()):
            history_iids = list(self.train_user_pos.get(uid, []))
            hist_indices = torch.tensor(history_iids, device=self.device)
            hist_locs = self.item_location[hist_indices]

            centroid = hist_locs.mean(dim=0)
            batch_centroids[i] = centroid

        return batch_centroids

    def compute_haversine_distance(self, center_locs):
        lat1 = center_locs[:, 0:1]
        lon1 = center_locs[:, 1:2]

        lat2 = self.item_location[:, 0:1].t()
        lon2 = self.item_location[:, 1:2].t()

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        R = 6371.0
        dist = R * c
        return dist

class LocationKNNRecommender(LocationRecommender):
    def compute_scores(self, users):
        user_centers = self.get_user_centroids(users)
        dists = self.compute_haversine_distance(user_centers)
        scores = -dists

        return scores

class LocationPopularityRecommender(LocationRecommender):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.distance_decay = 1.0
        self.epsilon = 0.1

    def compute_scores(self, users):
        user_centers = self.get_user_centroids(users)
        dists = self.compute_haversine_distance(user_centers)

        pop_scores = self.item_popularity.unsqueeze(0)
        scores = pop_scores / (dists + self.epsilon)
        return scores

class EmbeddingRecommender(Recommender):
    def load_embeddings(self, name):
        model_path = f"{self.cfg.paths.embedding}/{name}.pt"

        user_embs, item_embs = torch.load(model_path, map_location="cpu")
        self.user_embs = user_embs.to(self.device)
        self.item_embs = item_embs.to(self.device)

    def get_item_size(self):
        return self.item_embs.shape[0]

class EmbeddingDotRecommender(EmbeddingRecommender):
    def compute_scores(self, users):
        batch_user_embs = self.user_embs[users]
        scores = torch.matmul(batch_user_embs, self.item_embs.t())
        return scores

class EmbeddingCosineRecommender(EmbeddingRecommender):
    def compute_scores(self, users):
        batch_user_embs = self.user_embs[users]

        norm_user_embs = F.normalize(batch_user_embs, p=2, dim=1)
        norm_item_embs = F.normalize(self.item_embs, p=2, dim=1)

        scores = torch.matmul(norm_user_embs, norm_item_embs.t())
        return scores

class EmbeddingDistanceRecommender(EmbeddingRecommender):
    def compute_scores(self, users):
        batch_user_embs = self.user_embs[users]

        dists = torch.cdist(batch_user_embs, self.item_embs, p=2)
        scores = -dists

        return scores

def load_recommender(cfg, type, name):
    recommender = None
    if type == "baseline":
        if name == "Popularity":
            recommender = LocationPopularityRecommender(cfg)
        elif name == "KNN":
            recommender = LocationKNNRecommender(cfg)
        else:
            raise NotImplementedError

    elif type == "dot":
        recommender = EmbeddingDotRecommender(cfg)
        recommender.load_embeddings(name)

    elif type == "cosine":
        recommender = EmbeddingCosineRecommender(cfg)
        recommender.load_embeddings(name)

    elif type == "distance":
        recommender = EmbeddingDistanceRecommender(cfg)
        recommender.load_embeddings(name)

    else:
        raise NotImplementedError

    return recommender