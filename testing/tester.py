import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import tqdm
from torch.utils.data import DataLoader

from testing.recommender import *

class Tester:
    def __init__(self, cfg, recommender):
        self.recommender = recommender
        self.rank = cfg.test.rank

        self.device = torch.device(cfg.test.device)
        self.user_loader, self.test_user_pos = self.prepare_dataset(cfg)

    def prepare_dataset(self, cfg):
        test_df = pd.read_csv(f"{cfg.paths.splits}/test.csv")

        test_users = list(test_df["uid"].unique())
        user_loader = DataLoader(test_users, batch_size=cfg.test.batch_size, shuffle=False)

        test_user_pos = test_df.groupby("uid")["iid"].apply(set).to_dict()

        return user_loader, test_user_pos

    def calculate_recall_ndcg(self, topk_indices, users):
        topk_indices = topk_indices.cpu().numpy()
        users = users.cpu().numpy()

        recall_sum, ndcg_sum = 0.0, 0.0

        idcg_denom = 1.0 / np.log2(np.arange(self.rank) + 2.0)

        for i, uid in enumerate(users):
            pred_items = topk_indices[i]
            target_items = self.test_user_pos.get(uid, set())

            if len(target_items) == 0:
                continue

            hits = [1.0 if item in target_items else 0.0 for item in pred_items]
            hits = np.array(hits)

            recall_sum += hits.sum() / len(target_items)

            dcg = (hits * idcg_denom).sum()

            num_pos = len(target_items)
            ideal_len = min(self.rank, num_pos)

            if ideal_len > 0:
                idcg = idcg_denom[:ideal_len].sum()
                ndcg_sum += dcg / idcg

        return recall_sum, ndcg_sum

    def test(self):
        recall_sum, ndcg_sum, n_users = 0.0, 0.0, 0

        with torch.no_grad():
            for batch_users in tqdm.tqdm(self.user_loader, desc="Evaluating"):
                batch_users = batch_users.to(self.device)

                topk_indices = self.recommender.recommend(batch_users, self.rank)
                recall, ndcg = self.calculate_recall_ndcg(topk_indices, batch_users)

                recall_sum += recall
                ndcg_sum += ndcg
                n_users += batch_users.size(0)

        metric = {"recall": recall_sum / n_users, "ndcg": ndcg_sum / n_users}
        return metric

def test_recommender(cfg, type, name):
    recommender = None
    if type == "baseline":
        if name == "popularity":
            pass

    elif type == "dot":
        recommender = EmbeddingDotRecommender(cfg)
        recommender.load_embeddings(name)

    else:
        raise NotImplementedError

    tester = Tester(cfg, recommender)
    metric = tester.test()

    return metric
