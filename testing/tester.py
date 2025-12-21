from collections import Counter

import numpy as np
import tqdm
from scipy.stats import spearmanr, entropy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from testing.recommender import *

class Tester:
    def __init__(self, cfg, recommender):
        self.recommender = recommender
        self.rank = cfg.test.rank

        self.device = torch.device(cfg.test.device)
        self.user_loader, self.test_user_pos = self.prepare_dataset(cfg)

        self.info_dict, self.counts_dict = self.compute_dataset_metrics(cfg)
        self.n_items = self.load_item_size(cfg)

    def prepare_dataset(self, cfg):
        test_df = load_split_dataset(cfg, "test")

        test_users = list(test_df["uid"].unique())
        user_loader = DataLoader(test_users, batch_size=cfg.test.batch_size, shuffle=False)

        test_user_pos = test_df.groupby("uid")["iid"].apply(set).to_dict()

        return user_loader, test_user_pos

    def compute_dataset_metrics(self, cfg):
        train_df = load_split_dataset(cfg, "train")
        item_counts = train_df["iid"].value_counts()
        total_interactions = len(train_df)

        info_dict, counts_dict = {}, {}
        for iid, count in item_counts.items():
            prob = count / total_interactions
            info_dict[iid] = -np.log2(prob)
            counts_dict[iid] = count

        return info_dict, counts_dict

    def load_item_size(self, cfg):
        with open(f"{cfg.paths.result}/dataset_size.txt", "r") as f:
            _, item_size = [int(l) for l in f.read().split(",")]

        return item_size

    def load_item_embeddings(self, indices):
        if hasattr(self.recommender, "item_embs"):
            item_embs = self.recommender.item_embs[indices]
            item_embs = F.normalize(item_embs, p=2, dim=2)
            return item_embs

        else:
            return None

    def calculate_item_metrics(self, topk_indices, users):
        topk_indices = topk_indices.cpu().numpy()
        users = users.cpu().numpy()

        idcg_denom = 1.0 / np.log2(np.arange(self.rank) + 2.0)

        recall_sum, ndcg_sum, novelty_sum = 0.0, 0.0, 0.0

        for i, uid in enumerate(users):
            pred_items = topk_indices[i]
            target_items = self.test_user_pos.get(uid, set())

            hits = [1.0 if item in target_items else 0.0 for item in pred_items]
            hits = np.array(hits)

            recall_sum += hits.sum() / len(target_items)

            dcg = (hits * idcg_denom).sum()
            ideal_len = min(self.rank, len(target_items))
            if ideal_len > 0:
                idcg = idcg_denom[:ideal_len].sum()
                ndcg_sum += dcg / idcg

            user_novelty = [self.info_dict.get(item, 0.0) for item in pred_items]
            novelty_sum += np.mean(user_novelty) if user_novelty else 0.0

        return recall_sum, ndcg_sum, novelty_sum

    def calculate_embedding_metrics(self, topk_indices):
        batch_embeddings = self.load_item_embeddings(topk_indices)

        diversity_sum = 0.0
        if batch_embeddings is not None:
            sim_matrix = torch.bmm(batch_embeddings, batch_embeddings.transpose(1, 2))
            dist_matrix = 1.0 - sim_matrix

            k = self.rank
            if k > 1:
                triu_indices = torch.triu_indices(k, k, offset=1)
                upper_tri_dists = dist_matrix[:, triu_indices[0], triu_indices[1]]
                diversity_sum = upper_tri_dists.mean(dim=1).sum().item()

        return diversity_sum

    def calculate_batch_metrics(self, topk_indices, users):
        recall_sum, ndcg_sum, novelty_sum = self.calculate_item_metrics(topk_indices, users)
        diversity_sum = self.calculate_embedding_metrics(topk_indices)

        return recall_sum, ndcg_sum, novelty_sum, diversity_sum

    def calculate_frequency_metrics(self, global_rec_counts):
        n_covered = len(global_rec_counts)
        coverage = n_covered / self.n_items

        counts = np.zeros(self.n_items)
        for iid, count in global_rec_counts.items():
            counts[iid] = count

        counts = np.sort(counts)
        n = counts.shape[0]
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * counts).sum() / (n * counts.sum())

        return coverage, gini

    def calculate_popularity_metrics(self, global_rec_counts):
        train_counts = np.zeros(self.n_items)
        rec_counts = np.zeros(self.n_items)

        for iid, count in self.counts_dict.items():
            if iid < self.n_items:
                train_counts[iid] = count

        for iid, count in global_rec_counts.items():
            if iid < self.n_items:
                rec_counts[iid] = count

        correlation, _ = spearmanr(train_counts, rec_counts)

        epsilon = 1e-10
        p_train = (train_counts + epsilon) / (train_counts.sum() + epsilon * self.n_items)
        q_rec   = (rec_counts + epsilon) / (rec_counts.sum() + epsilon * self.n_items)

        kl_div = entropy(p_train, q_rec)

        sorted_indices = np.argsort(train_counts)[::-1]
        n_head = int(self.n_items * 0.2)
        head_indices = sorted_indices[:n_head]
        tail_indices = sorted_indices[n_head:]

        head_pop_share = p_train[head_indices].sum()
        head_rec_share = q_rec[head_indices].sum()

        tail_pop_share = p_train[tail_indices].sum()
        tail_rec_share = q_rec[tail_indices].sum()

        head_prr = head_rec_share / head_pop_share
        tail_prr = tail_rec_share / tail_pop_share

        return correlation, kl_div, head_prr, tail_prr

    def test(self):
        recall_sum, ndcg_sum, novelty_sum, diversity_sum = 0.0, 0.0, 0.0, 0.0
        n_users = 0

        global_rec_counts = Counter()

        with torch.no_grad():
            for batch_users in tqdm.tqdm(self.user_loader, desc="Evaluating"):
                batch_users = batch_users.to(self.device)

                topk_indices = self.recommender.recommend(batch_users, self.rank)

                rec_list_flat = topk_indices.cpu().numpy().flatten()
                global_rec_counts.update(rec_list_flat)

                recall, ndcg, novelty, diversity = self.calculate_batch_metrics(topk_indices, batch_users)

                recall_sum += recall
                ndcg_sum += ndcg
                novelty_sum += novelty
                diversity_sum += diversity
                n_users += batch_users.size(0)

        coverage, gini = self.calculate_frequency_metrics(global_rec_counts)
        correlation, kl_div, head_prr, tail_prr = self.calculate_popularity_metrics(global_rec_counts)

        metrics = {
            "recall": recall_sum / n_users,
            "ndcg": ndcg_sum / n_users,
            "novelty": novelty_sum / n_users,
            "diversity": diversity_sum / n_users,
            "coverage": coverage,
            "gini": gini,
            "pop_correlation": correlation,
            "kl_divergence": kl_div,
            "head_prr": head_prr,
            "tail_prr": tail_prr
        }

        return metrics

def test_recommender(cfg, type, name):
    recommender = load_recommender(cfg, type, name)
    tester = Tester(cfg, recommender)
    print(f"    Start evaluating {name}")
    metric = tester.test()

    return metric
