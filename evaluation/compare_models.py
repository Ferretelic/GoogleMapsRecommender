import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hydra import initialize, compose
from omegaconf import OmegaConf

def update_performance_csv(cfg):
    metrics = []
    for log_file in os.listdir(f"{cfg.paths.logs}"):
        with open(f"{cfg.paths.logs}/{log_file}", "r") as f:
            results = json.load(f)

        recall = results["valid"]["recall"]
        ndcg = results["valid"]["ndcg"]

        min_index = np.argmax(ndcg)
        metrics.append((log_file.replace(".log", ""),
             recall[min_index], ndcg[min_index], min_index+1
        ))

    df = pd.DataFrame(metrics, columns=["name", "recall", "ndcg", "epoch"])
    df = df.sort_values(by="ndcg").reset_index(drop=True)

    print(df)

    df.to_csv(f"{cfg.paths.result}/performances.csv", index=False)

def load_hydra_config(name, config_path):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=name, overrides=[])
        return cfg

def plot_comparisons(names, targets, config_path, target_name=None):
    if target_name is None:
        target_name = targets[-1]

    sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.cubehelix_palette(n_colors=len(names), start=2.8, rot=0.1, dark=0.2, light=0.8)

    plt.suptitle(f"Comparison of performances of models based on {targets[-1]}")
    results = []
    for name in names:
        cfg = load_hydra_config(name, config_path)
        target = cfg[targets[0]][targets[1]]

        with open(f"../logs/{cfg.dataset.country}/{name}.json", "r") as f:
            ndcg = json.load(f)["valid"]["ndcg"]

        df = pd.DataFrame({"epoch": range(len(ndcg)), "ndcg": ndcg, "target": [target] * len(ndcg)})
        results.append(df)

    results = pd.concat(results, axis=0)
    sns.lineplot(results, x="epoch", y="ndcg", ax=ax1, hue="target", palette=palette)

    ax1.set_title("Comparison of training history of NDCG")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NDCG")
    ax1.legend()

    results = results[["target", "ndcg"]].groupby("target").agg("max")
    sns.barplot(results, x="target", y="ndcg", ax=ax2, hue="target", palette=palette)
    ax2.set_ylim(0.10, 0.13)
    ax2.set_title("Comparison of best valid NDCG")
    ax2.set_xlabel(target_name)
    ax2.set_ylabel("NDCG")
    ax2.legend()

    folder_path = f"../results/{cfg.dataset.country}/plots/comparisons/"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f"{folder_path}/{target_name}.png")

if __name__ == "__main__":
    config_path = "../config"

    names = ["emb_32", "baseline", "emb_128", "emb_256", "emb_512", "emb_1024"]
    targets = ["model", "embedding_dim"]
    plot_comparisons(names, targets, config_path)

    names = ["n_layers_2", "baseline", "n_layers_4", "n_layers_5", "n_layers_6"]
    targets = ["model", "n_layers"]
    plot_comparisons(names, targets, config_path)

    names = ["reg_0.01", "reg_0.001", "baseline", "reg_0.00001"]
    targets = ["training", "reg_weight"]
    plot_comparisons(names, targets, config_path)

    names = ["baseline", "batch_2048", "batch_4096"]
    targets = ["training", "batch_size"]
    plot_comparisons(names, targets, config_path)

    names = ["lr_0.01", "baseline", "lr_0.0001"]
    targets = ["training", "lr"]
    plot_comparisons(names, targets, config_path)

    names = ["emb_512_n_layers_4_reg_0.001", "emb_512_n_layers_5_reg_0.001", "emb_512_n_layers_6_reg_0.001"]
    targets = ["model", "n_layers"]
    plot_comparisons(names, targets, config_path, target_name="emb_512_reg_0.001_n_layers")
