import os

import pandas as pd
import numpy as np

from testing.recommender import *
from utils import *

def sample_test_users(cfg):
    df = load_splits_dataset(cfg)
    test_df = load_split_dataset(cfg, "test")

    test_users = test_df["uid"].unique()

    df = df[df["uid"].isin(test_users)]
    user_counts = df["uid"].value_counts().reset_index()
    user_pools = user_counts[user_counts["count"] >= cfg.inference.min_num_reviews]["uid"].values

    sample_users = np.random.choice(user_pools, replace=False, size=cfg.inference.num_samples)

    return sample_users

class UserSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        mappings = load_mappings(cfg)
        self.index2gmap = mappings["index2gmap"]
        self.index2user = mappings["index2user"]
        self.gmap2index = mappings["gmap2index"]

        self.gmap2info = self.load_item_information()

    def load_item_information(self):
        meta = load_combined_datast(self.cfg, "meta")
        meta = meta[["gmap_id", "name", "state", "address", "category"]]

        gmap2info = {gmap_id: (name, state, address, category) for (gmap_id, name, state, address, category) in meta.values}
        return gmap2info

    def get_user_recommendations(self, topk_scores, topk_indices):
        items = []

        topk_gmaps = [self.index2gmap[str(item)] for item in topk_indices]
        for index in range(topk_scores.shape[0]):
            gmap_id = topk_gmaps[index]
            score = topk_scores[index]
            iid = topk_indices[index]

            name, state, address, category = self.gmap2info[gmap_id]

            item_info = {
                "rank": index + 1,
                "iid": int(iid),
                "gmap_id": str(gmap_id),
                "score": float(score),
                "name": name,
                "state": state,
                "address": address,
                "category": category
            }
            items.append(item_info)

        return items

    def log_recommendation(self, results):
        model_info = results["model"]
        user_info = results["user"]
        user_history = results["history"]
        user_recommendations = results["recommendations"]

        print("=" * 80)
        print(f"Recommendations for user {user_info["name"]} with {model_info["name"]}")
        print("  History")
        for item in user_history:
            print(f"    [{item["iid"]:5d}] [{item["state"]:15s}] {item["name"]}")

        print("-" * 80)
        print("  Recommendations")
        for item in user_recommendations:
            print(f"    [{item["iid"]:5d}] {item["rank"]:2d} [{item["state"]:15s}] {item["name"]} | {item["score"]:.3f}")

        print("=" * 80)
        print()

class TestUserSampler(UserSampler):
    def __init__(self, cfg, sample_users):
        super().__init__(cfg)

        self.users = torch.tensor(sample_users, dtype=torch.long)
        self.user2name = self.load_user_name()
        self.user2history = self.load_user_history()

    def load_user_name(self):
        review = load_combined_datast(self.cfg, "review")[["user_id", "name"]]

        user2name = {user_id: name for (user_id, name) in review.values}
        return user2name

    def load_user_history(self):
        df = load_splits_dataset(self.cfg)
        df["date"] = pd.to_datetime(df["time"], unit="ms").dt.strftime("%Y/%m/%d")
        df = df.sort_values(by=["uid", "time"])

        df["item_dict"] = [
            {
                "iid": item,
                "unix_time": time,
                "date": date,
                "split": split,
            }
            for item, time, date, split in zip(df["iid"], df["time"], df["date"], df["split"])
        ]
        user2history = df.groupby("uid")["item_dict"].apply(list).to_dict()

        return user2history

    def get_user_information(self, index):
        user = self.users[index].item()
        user_id = self.index2user[str(user)]
        user_name = self.user2name[user_id]

        user_info = {"uid": int(user), "user_id": int(user_id), "name": user_name}

        return user_info

    def run_recommender(self, model_info):
        recommender = load_recommender(self.cfg, model_info)
        users = self.users.to(self.device)

        scores = recommender.compute_scores(users)
        scores = recommender.mask_train_items(scores, users)

        topk_items = torch.topk(scores, k=self.cfg.inference.topk, dim=1)
        topk_scores = topk_items.values.detach().cpu().numpy()
        topk_indices = topk_items.indices.detach().cpu().numpy()

        return topk_scores, topk_indices

    def get_user_history(self, user_info):
        user_hisotry = self.user2history[user_info["uid"]]

        history = []
        for item_dict in user_hisotry:
            gmap_id = self.index2gmap[str(item_dict["iid"])]
            name, state, address, category = self.gmap2info[gmap_id]

            item_dict["gmap_id"] = gmap_id
            item_dict["name"] = name
            item_dict["state"] = state
            item_dict["address"] = address
            item_dict["category"] = category
            history.append(item_dict)

        return history

    def sample(self, model_info, log):
        topk_scores, topk_indices = self.run_recommender(model_info)

        results_path = f"{self.cfg.paths.result}/samples/{model_info["model"]}/{model_info["type"]}"
        os.makedirs(results_path, exist_ok=True)

        print(f"  Start running recommender {model_info["name"]}")

        for index in range(self.users.size(0)):
            user_info = self.get_user_information(index)
            user_history = self.get_user_history(user_info)
            user_recommendations = self.get_user_recommendations(topk_scores[index], topk_indices[index])

            results = {"user": user_info, "model": model_info, "history": user_history, "recommendations": user_recommendations}

            with open(f"{results_path}/{user_info["user_id"]}.json", "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

            if (user_info["user_id"] in log["user_ids"] and
                model_info["model"] == log["model"] and
                    model_info["type"] == log["type"]):
                self.log_recommendation(results)

class NewUserSampler(UserSampler):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_user_information(self, user):
        return {"name": user["name"], "user_id": user["user_id"]}

    def compute_new_user_scores(self, recommender, user_emb, type):
        if type == "dot":
            scores = torch.matmul(user_emb, recommender.item_embs.t())

        elif type == "cosine":
            norm_user_embs = F.normalize(user_emb, p=2, dim=1)
            norm_item_embs = F.normalize(recommender.item_embs, p=2, dim=1)
            scores = torch.matmul(norm_user_embs, norm_item_embs.t())

        elif type == "distance":
            dists = torch.cdist(user_emb, recommender.item_embs, p=2)
            scores = -dists

        else:
            raise NotImplementedError

        return scores

    def run_recommender(self, model_info, user):
        recommender = load_recommender(self.cfg, model_info)
        iids = [self.gmap2index[gmap_id] for gmap_id in user["gmap_ids"]]
        iids = torch.tensor(iids, dtype=torch.long).to(self.device)

        user_emb = torch.mean(recommender.item_embs[iids], dim=0, keepdim=True)
        scores = self.compute_new_user_scores(recommender, user_emb, model_info["type"])
        scores[:, iids] = -float("inf")

        topk_items = torch.topk(scores, k=self.cfg.inference.topk, dim=1)
        topk_scores = topk_items.values.detach().cpu().numpy()
        topk_indices = topk_items.indices.detach().cpu().numpy()

        return topk_scores, topk_indices

    def get_user_history(self, user):
        history = []
        for gmap_id in user["gmap_ids"]:
            item_dict = {}

            iid = self.gmap2index[gmap_id]
            name, state, address, category = self.gmap2info[gmap_id]

            item_dict["iid"] = iid
            item_dict["gmap_id"] = gmap_id
            item_dict["name"] = name
            item_dict["state"] = state
            item_dict["address"] = address
            item_dict["category"] = category
            history.append(item_dict)

        return history

    def sample(self, model_info, users):
        for user in users:
            user_info = self.get_user_information(user)
            topk_scores, topk_indices = self.run_recommender(model_info, user)

            results_path = f"{self.cfg.paths.result}/new_users/{model_info["model"]}/{model_info["type"]}"
            os.makedirs(results_path, exist_ok=True)
            file_path = f"{results_path}/{user_info["user_id"]}.json"

            user_history = self.get_user_history(user)
            user_recommendations = self.get_user_recommendations(topk_scores[0], topk_indices[0])

            results = {"user": user_info, "model": model_info, "history": user_history, "recommendations": user_recommendations}

            with open(file_path, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

            self.log_recommendation(results)