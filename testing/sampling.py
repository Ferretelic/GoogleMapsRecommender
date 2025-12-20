import os

import pandas as pd
import numpy as np

from testing.recommender import *

def load_dataset(cfg):
    train_df = pd.read_csv(f"{cfg.paths.splits}/train.csv")
    train_df["split"] = train_df["uid"].apply(lambda x: "train")

    valid_df = pd.read_csv(f"{cfg.paths.splits}/valid.csv")
    valid_df["split"] = valid_df["uid"].apply(lambda x: "valid")

    test_df = pd.read_csv(f"{cfg.paths.splits}/test.csv")
    test_df["split"] = test_df["uid"].apply(lambda x: "test")

    df = pd.concat([train_df, valid_df, test_df], axis=0)

    return test_df, df

def sample_test_users(cfg):
    test_df, df = load_dataset(cfg)
    test_users = test_df["uid"].unique()

    df = df[df["uid"].isin(test_users)]
    user_counts = df["uid"].value_counts().reset_index()
    user_pools = user_counts[user_counts["count"] >= cfg.inference.min_num_reviews]["uid"].values

    sample_users = np.random.choice(user_pools, replace=False, size=cfg.inference.num_samples)

    return sample_users

class Sampler():
    def __init__(self, cfg, sample_users):
        self.users = torch.tensor(sample_users, dtype=torch.long)
        self.cfg = cfg

        self.index2gmap, self.index2user = self.load_mappings()

        self.gmap2info = self.load_item_information()
        self.user2name = self.load_user_name()
        self.user2history = self.load_user_history()

    def load_mappings(self):
        with open(f"{self.cfg.paths.combined}/mappings.json", "r") as f:
            mappings = json.load(f)

        return mappings["index2gmap"], mappings["index2user"]

    def load_item_information(self):
        meta = pd.read_csv(f"{self.cfg.paths.combined}/meta.csv")[["gmap_id", "name", "state"]]

        gmap2info = {gmap_id: (name, state) for (gmap_id, name, state) in meta.values}
        return gmap2info

    def load_user_name(self):
        review = pd.read_csv(f"{self.cfg.paths.combined}/review.csv")[["user_id", "name"]]

        user2name = {user_id: name for (user_id, name) in review.values}
        return user2name

    def load_user_history(self):
        _, df = load_dataset(self.cfg)
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

    def get_model_information(self, model):
        name, type, model_name = model
        model_info = {"name": name, "type": type, "model_name": model_name}

        return model_info

    def run_recommender(self, model_info):
        recommender = load_recommender(self.cfg, model_info["type"], model_info["model_name"])
        users = self.users.to(recommender.device)

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
            name, state = self.gmap2info[gmap_id]

            item_dict["gmap_id"] = gmap_id
            item_dict["name"] = name
            item_dict["state"] = state
            history.append(item_dict)

        return history

    def get_user_recommendations(self, topk_scores, topk_indices):
        items = []

        topk_gmaps = [self.index2gmap[str(item)] for item in topk_indices]
        for index in range(topk_scores.shape[0]):
            gmap_id = topk_gmaps[index]
            score = topk_scores[index]
            iid = topk_indices[index]

            name, state = self.gmap2info[gmap_id]

            item_info = {"rank": index + 1, "iid": int(iid), "gmap_id": str(gmap_id), "score": float(score), "name": name, "state": state}
            items.append(item_info)

        return items

    def sample(self, model):
        model_info = self.get_model_information(model)
        topk_scores, topk_indices = self.run_recommender(model_info)

        results_path = f"{self.cfg.paths.result}/samples/{model_info["model_name"]}/{model_info["type"]}"
        os.makedirs(results_path, exist_ok=True)

        for index in range(self.users.size(0)):
            user_info = self.get_user_information(index)
            user_history = self.get_user_history(user_info)
            user_recommendations = self.get_user_recommendations(topk_scores[index], topk_indices[index])

            results = {"user": user_info, "model": model_info, "history": user_history, "recommendations": user_recommendations}

            with open(f"{results_path}/{user_info["user_id"]}.json", "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

    def print_recommendation(self, results):
        user_info = results["user_info"]
        user_hisotry = results["history"]
        user_recommendations = results["recommendations"]

        print("=" * 80)
        print(f"Recommendations for user {user_info["name"]} [{user_info["uid"]}]")
        print("  History")
        for item in user_history:
            print(f"    [{item["iid"]:5d}] {item["date"]} [{item["state"]:15s}] ({item["split"]:5s}) {item["name"]}")

        print("-" * 80)
        print("  Recommendations")
        for item in user_recommendations:
            print(f"    [{item["iid"]:5d}] {item["rank"]:2d} [{item["state"]:15s}] {item["name"]} | {item["score"]:.3f}")

        print("=" * 80)
        print()
