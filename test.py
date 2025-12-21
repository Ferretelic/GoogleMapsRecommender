import os

import hydra
from omegaconf import OmegaConf
import pandas as pd

from testing.tester import *
from testing.sampler import *
from testing.new_user import *
from testing.metrics import *

from utils import *

def load_models(cfg):
    models = []
    print("Testing baselines...")
    for baseline in cfg.baselines:
        models.append({"name": baseline, "type": "baseline", "model": baseline})

    print("Testing trained embeddings...")
    for model_info in cfg.embeddings:
        models.append(OmegaConf.to_container(model_info, resolve=True))

    return models

def calculate_metrics(cfg):
    if os.path.exists(f"{cfg.paths.result}/test_performances.csv"):
        return

    metrics = []
    for model_info in load_models(cfg):
        metric = test_recommender(cfg, model_info)
        metric = model_info | metric
        metrics.append(metric)

    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(f"{cfg.paths.result}/test_performances.csv", index=False)

    plot_test_results(cfg, df)

def run_inference(cfg):
    sample_users = sample_test_users(cfg)
    sampler = TestUserSampler(cfg, sample_users)

    for model_info in load_models(cfg):
        sampler.sample(model_info, cfg.inference.log)

def recommend_new_users(cfg):
    users = load_new_users(cfg)
    sampler = NewUserSampler(cfg)

    for model_info in load_models(cfg):
        sampler.sample(OmegaConf.to_container(model_info, resolve=True), users)

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg):
    seed_everything()

    print("Comparing model performances on validation datast...")
    update_valid_performances(cfg)

    print("Adding new user for inference...")
    add_new_user(cfg)

    print("Calculating metrics on test dataset...")
    calculate_metrics(cfg)

    print("Running inference on sampled users...")
    run_inference(cfg)

    print("Running recommenders on new users...")
    recommend_new_users(cfg)


if __name__ == "__main__":
    main()