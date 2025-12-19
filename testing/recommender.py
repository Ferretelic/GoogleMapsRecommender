import os

import torch
import pandas as pd

class Recommender():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.test.device)

        train_df = pd.read_csv(f"{cfg.paths.splits}/train.csv")
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

        return scores

    def get_top_k_items(self, scores, rank):
        _, topk_indices = torch.topk(scores, k=rank, dim=1)
        return topk_indices

    def recommend(self, users, rank):
        scores = self.compute_scores(users)
        scores = self.mask_train_items(scores, users)
        topk_indices = self.get_top_k_items(scores, rank)

        return topk_indices

class EmbeddingRecommender(Recommender):
    def load_embeddings(self, name):
        model_path = f"{self.cfg.paths.embedding}/{name}.pt"

        user_embs, item_embs = torch.load(model_path)
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
        scores = torch.matmul(batch_user_embs, self.item_embs.t())
        return scores
