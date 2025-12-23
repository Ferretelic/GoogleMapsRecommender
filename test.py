import os

import hydra
from omegaconf import OmegaConf
import pandas as pd

from testing.tester import *
from testing.sampler import *
from testing.new_user import *
from testing.metrics import *
from testing.dim_reduction import *

from utils import *

def load_models(cfg):
    models = []
    for model_info in cfg.baselines:
        models.append(OmegaConf.to_container(model_info, resolve=True))

    for model_info in cfg.embeddings:
        models.append(OmegaConf.to_container(model_info, resolve=True))

    return models

def calculate_valid_performances(cfg):
    update_valid_performances(cfg)
    plot_validation_results(cfg)

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
        sampler.sample(model_info, users, cfg.inference.log)

def analyze_models(cfg):
    models = []
    for model_info in load_models(cfg):
        if model_info["type"] == "dot":
            models.append(model_info["model"])

    for model in models:
        analyze_embeddings(cfg, model)

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg):
    seed_everything()

    print("Plotting training history...")
    plot_training_history(cfg)

    print("Comparing model performances on validation datast...")
    calculate_valid_performances(cfg)

    # print("Analysing embedding vectors...")
    # analyze_models(cfg)

    # print("Calculating metrics on test dataset...")
    # calculate_metrics(cfg)

    # print("Running inference on test users...")
    # run_inference(cfg)

    # print("Adding new user for inference...")
    # if cfg.inference.add_user:
    #     add_new_user(cfg)

    # print("Running recommenders on new users...")
    # recommend_new_users(cfg)


if __name__ == "__main__":
    main()