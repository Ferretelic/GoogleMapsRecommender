import torch
import torch.nn.functional as F
import numpy as np

from utils import *

class Recommender:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        train_df = load_split_dataset(cfg, "train")
        self.train_df = train_df.sort_values(by=["uid", "time"]).reset_index(drop=True)
        self.train_user_pos = self.train_df.groupby("uid")["iid"].apply(list).to_dict()

        _, self.item_size = load_dataset_sizes(cfg)

    def mask_test_user_train_items(self, scores, users):
        users = users.cpu().numpy()
        rows, cols = [], []
        for i, uid in enumerate(users):
            train_items = self.train_user_pos[uid]
            train_items = list(train_items)
            if train_items:
                rows.extend([i] * len(train_items))
                cols.extend(train_items)

        if rows:
            scores[rows, cols] = -float("inf")

        return scores

    def mask_new_user_train_items(self, scores, iids):
        scores[:, iids] = -float("inf")
        return scores

    def get_top_k_items(self, scores, rank):
        _, topk_indices = torch.topk(scores, k=rank, dim=1)
        return topk_indices

    def recommend(self, users, rank):
        scores = self.compute_test_user_scores(users)
        scores = self.mask_test_user_train_items(scores, users)
        topk_indices = self.get_top_k_items(scores, rank)

        return topk_indices

    def get_test_user_scores(self, users):
        scores = self.compute_test_user_scores(users)
        scores = self.mask_test_user_train_items(scores, users)

        return scores

    def get_new_user_scores(self, iids):
        scores = self.compute_new_user_scores(iids)
        scores = self.mask_new_user_train_items(scores, iids)

        return scores

class LocationRecommender(Recommender):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.item_popularity = self.load_item_popularity(cfg)
        self.item_location = self.load_item_location(cfg)

    def load_item_popularity(self, cfg):
        item_popularity = torch.zeros(self.item_size, dtype=torch.float32).to(self.device)

        for iid, count in self.train_df["iid"].value_counts().reset_index().values:
            item_popularity[iid] = count

        return item_popularity

    def load_item_location(self, cfg):
        item_location = torch.zeros((self.item_size, 2), dtype=torch.float32)
        df = load_splits_dataset(cfg)
        df = df.groupby("iid")[["latitude", "longitude"]].first().reset_index()

        for (iid, latitude, longitude) in df.values:
            item_location[int(iid), :] = torch.tensor(np.radians((latitude, longitude)), dtype=torch.float32)

        item_location = item_location.to(self.device)

        return item_location

    def get_test_user_centroids(self, users):
        batch_centroids = torch.zeros(users.size(0), 2, device=self.device)

        for i, uid in enumerate(users.cpu().numpy()):
            history_iids = list(self.train_user_pos[uid])
            hist_indices = torch.tensor(history_iids, device=self.device)
            hist_locs = self.item_location[hist_indices]

            centroid = hist_locs.mean(dim=0)
            batch_centroids[i] = centroid

        return batch_centroids

    def get_new_user_centroids(self, iids):
        hist_locs = self.item_location[iids]
        centroid = hist_locs.mean(dim=0, keepdim=True)

        return centroid

    def get_test_user_latest_locations(self, users):
        latest_iids = []
        for uid in users.cpu().numpy():
            latest_iids.append(self.train_user_pos[uid][-1])

        latest_iids = torch.tensor(latest_iids, device=self.device)
        latest_locs = self.item_location[latest_iids]

        return latest_locs

    def get_new_user_latest_locations(self, iids):
        user_loc = self.item_location[iids[-1].unsqueeze(0)]

        return user_loc

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
    def compute_test_user_scores(self, users):
        user_centers = self.get_test_user_centroids(users)
        dists = self.compute_haversine_distance(user_centers)
        scores = -dists

        return scores

    def compute_new_user_scores(self, iids):
        user_center = self.get_new_user_centroids(iids)
        dists = self.compute_haversine_distance(user_center)
        scores = -dists

        return scores

class LocationLatestKNNRecommender(LocationRecommender):
    def compute_test_user_scores(self, users):
        user_locs = self.get_test_user_latest_locations(users)
        dists = self.compute_haversine_distance(user_locs)
        scores = -dists

        return scores

    def compute_new_user_scores(self, iids):
        user_loc = self.get_new_user_latest_locations(iids)
        dists = self.compute_haversine_distance(user_loc)
        scores = -dists

        return scores

class LocationPopularityRecommender(LocationRecommender):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.distance_decay = 1.0
        self.epsilon = 0.1

    def compute_test_user_scores(self, users):
        user_centers = self.get_test_user_centroids(users)
        dists = self.compute_haversine_distance(user_centers)

        pop_scores = self.item_popularity.unsqueeze(0)
        scores = pop_scores / (dists + self.epsilon)
        return scores

    def compute_new_user_scores(self, iids):
        user_center = self.get_new_user_centroids(iids)
        dists = self.compute_haversine_distance(user_center)

        pop_scores = self.item_popularity.unsqueeze(0)
        scores = pop_scores / (dists + self.epsilon)
        return scores

class LocationLatestPopularityRecommender(LocationRecommender):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.distance_decay = 1.0
        self.epsilon = 0.1

    def compute_test_user_scores(self, users):
        user_locs = self.get_test_user_latest_locations(users)
        dists = self.compute_haversine_distance(user_locs)

        pop_scores = self.item_popularity.unsqueeze(0)
        scores = pop_scores / (dists + self.epsilon)
        return scores

    def compute_new_user_scores(self, iids):
        user_loc = self.get_new_user_latest_locations(iids)
        dists = self.compute_haversine_distance(user_loc)

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
    def compute_test_user_scores(self, users):
        batch_user_embs = self.user_embs[users]
        scores = torch.matmul(batch_user_embs, self.item_embs.t())
        return scores

    def compute_new_user_scores(self, iids):
        user_emb = torch.mean(self.item_embs[iids], dim=0, keepdim=True)
        scores = torch.matmul(user_emb, self.item_embs.t())
        return scores

class EmbeddingFoldRecommender(EmbeddingRecommender):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.item_degrees = self.compute_item_degrees()

    def compute_test_user_scores(self, users):
        rows, cols, n_histories = [], [], []

        for i, uid in enumerate(users.cpu().numpy()):
            history_iids = self.train_user_pos.get(uid, [])
            if history_iids:
                length = len(history_iids)
                rows.extend([i] * length)
                cols.extend(history_iids)
                n_histories.extend([length] * length)

        rows_t = torch.tensor(rows, device=self.device, dtype=torch.long)
        cols_t = torch.tensor(cols, device=self.device, dtype=torch.long)
        n_hist_t = torch.tensor(n_histories, device=self.device, dtype=torch.float32)

        target_item_degrees = self.item_degrees[cols_t]
        norm_values = 1.0 / torch.sqrt(n_hist_t * target_item_degrees)

        indices = torch.stack([rows_t, cols_t])
        shape = (len(users), self.item_size)

        user_history_graph = torch.sparse_coo_tensor(indices, norm_values, size=shape, device=self.device)
        batch_user_embs = torch.sparse.mm(user_history_graph, self.item_embs)
        scores = torch.matmul(batch_user_embs, self.item_embs.t())

        return scores

    def compute_item_degrees(self):
        degrees = torch.zeros(self.item_size, device=self.device)

        item_counts = self.train_df["iid"].value_counts()
        indices = torch.tensor(item_counts.index.values, device=self.device)
        counts = torch.tensor(item_counts.values, dtype=torch.float32, device=self.device)

        degrees[indices] = counts
        degrees[degrees == 0] = 1.0
        return degrees

    def compute_new_user_scores(self, iids):
        n_history = iids.size(0)
        target_item_degrees = self.item_degrees[iids]

        norm_factor = 1.0 / torch.sqrt(n_history * target_item_degrees)
        target_item_embs = self.item_embs[iids]

        user_emb = torch.sum(norm_factor.unsqueeze(1) * target_item_embs, dim=0, keepdim=True)
        scores = torch.matmul(user_emb, self.item_embs.t())

        return scores

class EmbeddingCosineRecommender(EmbeddingRecommender):
    def compute_test_user_scores(self, users):
        batch_user_embs = self.user_embs[users]

        norm_user_embs = F.normalize(batch_user_embs, p=2, dim=1)
        norm_item_embs = F.normalize(self.item_embs, p=2, dim=1)

        scores = torch.matmul(norm_user_embs, norm_item_embs.t())
        return scores

    def compute_new_user_scores(self, iids):
        user_emb = torch.mean(self.item_embs[iids], dim=0, keepdim=True)

        norm_user_embs = F.normalize(user_emb, p=2, dim=1)
        norm_item_embs = F.normalize(self.item_embs, p=2, dim=1)

        scores = torch.matmul(norm_user_embs, norm_item_embs.t())
        return scores

class EmbeddingDistanceRecommender(EmbeddingRecommender):
    def compute_test_user_scores(self, users):
        batch_user_embs = self.user_embs[users]

        dists = torch.cdist(batch_user_embs, self.item_embs, p=2)
        scores = -dists

        return scores

    def compute_new_user_scores(self, iids):
        user_emb = torch.mean(self.item_embs[iids], dim=0, keepdim=True)

        dists = torch.cdist(user_emb, self.item_embs, p=2)
        scores = -dists

        return scores

def load_recommender(cfg, model_info):
    if model_info["type"] == "centroid":
        if model_info["model"] == "popularity":
            recommender = LocationPopularityRecommender(cfg)
        elif model_info["model"] == "knn":
            recommender = LocationKNNRecommender(cfg)
        else:
            raise NotImplementedError

    elif model_info["type"] == "latest":
        if model_info["model"] == "popularity":
            recommender = LocationLatestPopularityRecommender(cfg)
        elif model_info["model"] == "knn":
            recommender = LocationLatestKNNRecommender(cfg)
        else:
            raise NotImplementedError

    elif model_info["type"] == "dot":
        recommender = EmbeddingDotRecommender(cfg)
        recommender.load_embeddings(model_info["model"])

    elif model_info["type"] == "cosine":
        recommender = EmbeddingCosineRecommender(cfg)
        recommender.load_embeddings(model_info["model"])

    elif model_info["type"] == "distance":
        recommender = EmbeddingDistanceRecommender(cfg)
        recommender.load_embeddings(model_info["model"])

    elif model_info["type"] == "fold":
        recommender = EmbeddingFoldRecommender(cfg)
        recommender.load_embeddings(model_info["model"])

    else:
        raise NotImplementedError

    return recommender