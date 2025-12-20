import os
import random

import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch

from testing.tester import *
from testing.plot_results import *
from testing.sampling import *

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

    sample_users = sample_test_users

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
    sampler = Sampler(cfg, sample_users)

    for baseline in cfg.inference.baselines:
        model = (baseline, "baseline", baseline)
        sampler.sample(model)

    for model in cfg.inference.embeddings:
        sampler.sample(model)

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig):
    seed_everything()

    print("Calculating metrics on test dataset...")
    calculate_metrics(cfg)

    print("Running inference on sampled users...")
    run_inference(cfg)


if __name__ == "__main__":
    main()