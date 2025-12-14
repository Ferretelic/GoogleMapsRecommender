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

    def calculate_scores(self, batch_users, user_embs, item_embs):
        u_embs = user_embs[batch_users]
        scores = torch.matmul(u_embs, item_embs.t())

        scores = scores.cpu().numpy()
        return scores

    def get_top_k_items(self, batch_users, scores):
        for i, uid in enumerate(batch_users):
            train_pos_items = list(self.train_user_pos.get(uid, []))
            if len(train_pos_items) > 0:
                scores[i, train_pos_items] = -np.inf

        topk_indices = np.argpartition(scores, -self.rank, axis=1)[:, -self.rank:]

        rows = np.arange(len(batch_users))[:, None]
        topk_scores = scores[rows, topk_indices]
        sorted_idx = np.argsort(-topk_scores, axis=1)
        topk_items = topk_indices[rows, sorted_idx]

        return topk_items

    def calculate_recall(self, pred_items, target_items):
        if len(target_items) == 0:
            return 0.0

        num_hits = sum([1 for item in pred_items if item in target_items])
        recall = num_hits / len(target_items)
        return recall

    def calculate_ndcg(self, pred_items, target_items):
        dcg, idcg = 0.0, 0.0

        for i, item in enumerate(pred_items):
            if item in target_items:
                dcg += 1.0 / np.log2(i + 2)

        num_targets = len(target_items)
        k = len(pred_items)
        ideal_len = min(num_targets, k)

        for i in range(ideal_len):
            idcg += 1.0 / np.log2(i + 2)

        if idcg > 0:
            return dcg / idcg
        else:
            return 0

    def evaluate(self, embeddings):
        user_embs, item_embs = embeddings

        recall, ndcg, n_users = 0, 0, 0
        for batch_users in tqdm.tqdm(self.user_loader, desc="Evaluating Model"):
            batch_users = batch_users.to(user_embs.device)

            scores = self.calculate_scores(batch_users, user_embs, item_embs)

            batch_users_np = batch_users.cpu().numpy()
            topk_items = self.get_top_k_items(batch_users_np, scores)

            for i, uid in enumerate(batch_users_np):
                pred_items = topk_items[i]
                target_items = self.test_user_pos[uid]

                recall += self.calculate_recall(pred_items, target_items)
                ndcg += self.calculate_ndcg(pred_items, target_items)
                n_users += len(batch_users)

        return recall / n_users, ndcg / n_users
