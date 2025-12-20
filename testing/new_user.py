
import os
import json

import pandas as pd
import numpy as np

from testing.recommender import *

def load_new_users(cfg):
    users = []
    for file_name in os.listdir(cfg.paths.users):
        with open(f"{cfg.paths.users}/{file_name}", "r") as f:
            user = json.load(f)

        users.append(user)

    return users

def load_dataset(cfg):
    train_df = pd.read_csv(f"{cfg.paths.splits}/train.csv")
    train_df["split"] = train_df["uid"].apply(lambda x: "train")

    valid_df = pd.read_csv(f"{cfg.paths.splits}/valid.csv")
    valid_df["split"] = valid_df["uid"].apply(lambda x: "valid")

    test_df = pd.read_csv(f"{cfg.paths.splits}/test.csv")
    test_df["split"] = test_df["uid"].apply(lambda x: "test")

    df = pd.concat([train_df, valid_df, test_df], axis=0)

    return test_df, df

class NewUserSampler():
    def __init__(self, cfg):
        self.cfg = cfg

        self.index2gmap, self.gmap2index = self.load_mappings()
        self.gmap2info = self.load_item_information()

    def load_mappings(self):
        with open(f"{self.cfg.paths.combined}/mappings.json", "r") as f:
            mappings = json.load(f)

        return mappings["index2gmap"], mappings["gmap2index"]

    def load_item_information(self):
        meta = pd.read_csv(f"{self.cfg.paths.combined}/meta.csv")[["gmap_id", "name", "state", "address", "category"]]

        gmap2info = {gmap_id: (name, state, address, category) for (gmap_id, name, state, address, category) in meta.values}
        return gmap2info

    def get_user_information(self, user):
        return {"name": user["name"], "user_id": user["user_id"]}

    def get_model_information(self, model):
        name, type, model_name = model
        model_info = {"name": name, "type": type, "model_name": model_name}

        return model_info

    def recommend_new_users(self, model_info, user):
        recommender = load_recommender(self.cfg, model_info["type"], model_info["model_name"])
        iids = [self.gmap2index[gmap_id] for gmap_id in user["gmap_ids"]]
        iids = torch.tensor(iids, dtype=torch.long).to(recommender.device)

        user_emb = torch.mean(recommender.item_embs[iids], dim=0, keepdim=True)

        if model_info["type"] == "dot":
            scores = torch.matmul(user_emb, recommender.item_embs.t())
        elif model_info["type"] == "cosine":
            norm_user_embs = F.normalize(user_emb, p=2, dim=1)
            norm_item_embs = F.normalize(recommender.item_embs, p=2, dim=1)
            scores = torch.matmul(norm_user_embs, norm_item_embs.t())
        else:
            dists = torch.cdist(user_emb, recommender.item_embs, p=2)
            scores = -dists

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

    def sample(self, model, users):
        for user in users:
            user["gmap_ids"] = [gmap_id for gmap_id in user["gmap_ids"] if gmap_id in self.gmap2index.keys()]
            user_info = self.get_user_information(user)
            model_info = self.get_model_information(model)
            topk_scores, topk_indices = self.recommend_new_users(model_info, user)

            results_path = f"{self.cfg.paths.result}/new_users/{model_info["model_name"]}/{model_info["type"]}"
            os.makedirs(results_path, exist_ok=True)
            file_path = f"{results_path}/{user_info["user_id"]}.json"

            user_history = self.get_user_history(user)
            user_recommendations = self.get_user_recommendations(topk_scores[0], topk_indices[0])

            results = {"user": user_info, "model": model_info, "history": user_history, "recommendations": user_recommendations}

            with open(file_path, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

            self.log_recommendation(results)

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