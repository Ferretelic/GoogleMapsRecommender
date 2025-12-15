import numpy as np
import tqdm
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
        user_embs = user_embs[users]
        scores = torch.matmul(user_embs, item_embs.t())
        return scores

    def mask_train_items(self, scores, users):
        rows, cols = [], []
        for i, uid in enumerate(users):
            train_items = self.train_user_pos.get(uid, [])
            if train_items:
                rows.extend([i] * len(train_items))
                cols.extend(train_items)

        if rows:
            scores[rows, cols] = -float("inf")

        return scores

    def get_top_k_items(self, scores):
        _, topk_indices = torch.topk(scores, k=self.rank, dim=1)
        return topk_indices

    def construct_target_matrix(self, users, num_items, device):
        batch_size = len(users)

        target_len = torch.zeros(batch_size, device=device)
        batch_target_matrix = torch.zeros((batch_size, num_items), device=device, dtype=torch.bool)

        batch_rows, batch_cols = [], []
        for i, uid in enumerate(users):
            test_items = self.test_user_pos.get(uid, [])
            if test_items:
                target_len[i] = len(test_items)
                batch_rows.extend([i] * len(test_items))
                batch_cols.extend(test_items)

        if batch_rows:
            batch_target_matrix[batch_rows, batch_cols] = True

        return batch_target_matrix, target_len

    def compute_hits(self, target_matrix, topk_indices):
        hits = torch.gather(target_matrix, 1, topk_indices).float()
        return hits

    def calculate_recall(self, hits, target_len):
        recall_batch = hits.sum(1) / (target_len + 1e-10)
        return recall_batch.sum().item()

    def calculate_ndcg(self, hits, target_len, idcg_denominator):
        batch_size = hits.size(0)
        device = hits.device

        dcg_batch = (hits * idcg_denominator).sum(1)
        k_expanded = torch.ones(batch_size, device=device) * self.rank
        ideal_len = torch.min(target_len, k_expanded).long()

        idcg_table = torch.cumsum(idcg_denominator, dim=0)
        idcg_vals = torch.zeros(batch_size, device=device)
        valid_mask = ideal_len > 0
        if valid_mask.any():
            idcg_vals[valid_mask] = idcg_table[ideal_len[valid_mask] - 1]

        ndcg_batch = dcg_batch / (idcg_vals + 1e-10)
        return ndcg_batch.sum().item()

    def evaluate(self, embeddings):
        user_embs, item_embs = embeddings
        user_embs = user_embs.detach()
        item_embs = item_embs.detach()

        num_items = item_embs.shape[0]
        device = user_embs.device

        recall_sum, ndcg_sum, n_users = 0.0, 0.0, 0

        k_tensor = torch.arange(self.rank, device=device).float()
        idcg_denominator = 1.0 / torch.log2(k_tensor + 2.0)

        with torch.no_grad():
            for batch_users in tqdm.tqdm(self.user_loader, desc="Evaluating Model"):
                batch_users = batch_users.to(device)
                batch_users_list = batch_users.cpu().tolist()
                batch_size = batch_users.size(0)

                scores = self.compute_scores(user_embs, item_embs, batch_users)
                scores = self.mask_train_items(scores, batch_users_list)
                topk_indices = self.get_top_k_items(scores)
                target_matrix, target_len = self.construct_target_matrix(batch_users_list, num_items, device)
                hits = self.compute_hits(target_matrix, topk_indices)

                recall = self.calculate_recall(hits, target_len)
                ndcg = self.calculate_ndcg(hits, target_len, idcg_denominator)

                recall_sum += recall
                ndcg_sum += ndcg
                n_users += batch_size

        recall = recall_sum / n_users
        ndcg = ndcg_sum / n_users

        return recall, ndcg
