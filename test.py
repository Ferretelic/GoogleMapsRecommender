import os
import random

import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch

from testing.tester import *
from testing.sampler import *
from testing.new_user import *
from testing.metrics import *

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def calculate_metrics(cfg):
    if os.path.exists(f"{cfg.paths.result}/test_results.csv"):
        return

    metrics = []

    print("Testing baselines...")
    for baseline in cfg.test.baselines:
        metric = test_recommender(cfg, "baseline", baseline)

        metric["type"] = "baseline"
        metric["name"] = baseline
        metrics.append(metric)

    print("Testing trained embeddings...")
    for (name, type, embedding) in cfg.test.embeddings:
        metric = test_recommender(cfg, type, embedding)

        metric["type"] = type
        metric["name"] = name
        metrics.append(metric)

    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(f"{cfg.paths.result}/test_results.csv", index=False)

    plot_test_results(cfg, df)

def run_inference(cfg):
    sample_users = sample_test_users(cfg)
    sampler = TestUserSampler(cfg, sample_users)

    for baseline in cfg.inference.baselines:
        model = (baseline, "baseline", baseline)
        sampler.sample(model, cfg.inference.log)

    for model in cfg.inference.embeddings:
        sampler.sample(model, cfg.inference.log)

def recommend_new_users(cfg):
    users = load_new_users(cfg)
    sampler = NewUserSampler(cfg)

    for model in cfg.inference.embeddings:
        sampler.sample(model, users)

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig):
    seed_everything()

    print("Comparing model performances on validation datast...")
    update_valid_performances(cfg)

    print("Adding new user for inference...")
    add_new_user(cfg)

    # print("Calculating metrics on test dataset...")
    # calculate_metrics(cfg)

    # print("Running inference on sampled users...")
    # run_inference(cfg)

    # print("Running recommenders on new users...")
    # recommend_new_users(cfg)


if __name__ == "__main__":
    main()