import json
import os

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

class Evaluator():
    def __init__(self, eval_cfg, mode, datasets):
        train_df = datasets["train"].dataset.df_pos
        self.train_user_pos = train_df.groupby("uid")["iid"].apply(set).to_dict()

        test_df = datasets[mode]
        self.test_user_pos = test_df.groupby("uid")["iid"].apply(set).to_dict()
        test_users = list(self.test_user_pos.keys())
        self.user_loader = DataLoader(test_users, batch_size=eval_cfg.batch_size, shuffle=False)

        self.rank = eval_cfg.rank

    def compute_scores(self, user_embs, item_embs, users):
        batch_user_embs = user_embs[users]
        scores = torch.matmul(batch_user_embs, item_embs.t())
        return scores

    def mask_train_items(self, scores, users_list):
        rows, cols = [], []
        for i, uid in enumerate(users_list):
            train_items = self.train_user_pos.get(uid, set())
            train_items = list(train_items)
            if train_items:
                rows.extend([i] * len(train_items))
                cols.extend(train_items)

        if rows:
            scores[rows, cols] = -float("inf")

        return scores

    def get_top_k_items(self, scores):
        _, topk_indices = torch.topk(scores, k=self.rank, dim=1)
        return topk_indices

    def calculate_metrics_cpu(self, topk_indices, users_list):
        recall_sum = 0.0
        ndcg_sum = 0.0

        idcg_denom = 1.0 / np.log2(np.arange(self.rank) + 2.0)

        for i, uid in enumerate(users_list):
            pred_items = topk_indices[i]
            target_items = self.test_user_pos.get(uid, set())

            if len(target_items) == 0:
                continue

            hits = [1.0 if item in target_items else 0.0 for item in pred_items]
            hits = np.array(hits)

            recall_sum += hits.sum() / len(target_items)

            dcg = (hits * idcg_denom).sum()
            k = len(pred_items)
            num_pos = len(target_items)
            ideal_len = min(k, num_pos)
            if ideal_len > 0:
                idcg = idcg_denom[:ideal_len].sum()
                ndcg_sum += dcg / idcg

        return recall_sum, ndcg_sum

    def evaluate(self, embeddings):
        user_embs, item_embs = embeddings
        user_embs = user_embs.detach()
        item_embs = item_embs.detach()

        device = user_embs.device

        recall_sum, ndcg_sum, n_users = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_users in tqdm.tqdm(self.user_loader, desc="Evaluating Model"):
                batch_users_device = batch_users.to(device)
                batch_users_np = batch_users.numpy()
                batch_size = batch_users.size(0)

                scores = self.compute_scores(user_embs, item_embs, batch_users_device)
                scores = self.mask_train_items(scores, batch_users_np)

                topk_indices = self.get_top_k_items(scores)
                top_indices_np = topk_indices.cpu().numpy()

                recall, ndcg = self.calculate_metrics_cpu(top_indices_np, batch_users_np)

                recall_sum += recall
                ndcg_sum += ndcg
                n_users += batch_size

        recall = recall_sum / n_users
        ndcg = ndcg_sum / n_users

        return recall, ndcg