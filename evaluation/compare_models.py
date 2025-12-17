import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hydra import initialize, compose

def update_performance_csv(cfg):
    metrics = []
    for log_file in os.listdir(f"{cfg.paths.logs}"):
        with open(f"{cfg.paths.logs}/{log_file}", "r") as f:
            results = json.load(f)

        recall = results["valid"]["recall"]
        ndcg = results["valid"]["ndcg"]

        n_patience = len(ndcg) - (np.argmax(ndcg) + 1)

        min_index = np.argmax(ndcg)
        metrics.append((log_file.replace(".json", ""),
             recall[min_index], ndcg[min_index], min_index+1, n_patience
        ))

    df = pd.DataFrame(metrics, columns=["name", "recall", "ndcg", "epoch", "n_patience"])
    df = df.sort_values(by="ndcg").reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    print(df)

    df.to_csv(f"{cfg.paths.result}/performances.csv", index=False)

def load_hydra_config(name, config_path):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=name, overrides=[])
        return cfg

def plot_history_line(results, ax, palette):
    sns.lineplot(results, x="epoch", y="ndcg", ax=ax, hue="target", palette=palette)

    ax.set_title("Comparison of training history of NDCG")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NDCG")
    ax.legend()

def plot_ndcg_bar(results, ax, palette, target_name, sub_target_name=None):
    if sub_target_name is None:
        results = results[["target", "ndcg"]].groupby("target").agg("max")
        sns.barplot(results, x="target", y="ndcg", ax=ax, hue="target", palette=palette)
    else:
        results = results[["main_target", "sub_target", "ndcg"]].groupby(["main_target", "sub_target"]).agg("max")
        sns.barplot(results, x="main_target", y="ndcg", ax=ax, hue="sub_target", palette=palette)

    min = math.floor(results["ndcg"].min() * 200) / 200
    max = math.ceil(results["ndcg"].max() * 200) / 200
    ax.set_ylim(min, max)

    ax.set_title("Comparison of best valid NDCG")
    ax.set_xlabel(target_name)
    ax.set_ylabel("NDCG")
    ax.legend()

def plot_comparisons_one_target(names, targets, config_path, target_name=None):
    if target_name is None:
        target_name = targets[-1]

    sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    palette = sns.cubehelix_palette(n_colors=len(names), start=2.6, rot=0.1, dark=0.2, light=0.8)

    plt.suptitle(f"Comparison of performances of models based on {target_name}")
    results = []
    for name in names:
        cfg = load_hydra_config(name, config_path)
        target = cfg[targets[0]].get(targets[1], 0.0)

        with open(f"../logs/{cfg.dataset.country}/{name}.json", "r") as f:
            ndcg = json.load(f)["valid"]["ndcg"]

        df = pd.DataFrame({"epoch": range(len(ndcg)), "ndcg": ndcg, "target": [target] * len(ndcg)})
        results.append(df)

    results = pd.concat(results, axis=0).reset_index(drop=True)
    plot_history_line(results, ax1, palette)
    plot_ndcg_bar(results, ax2, palette, target_name)

    folder_path = f"../results/{cfg.dataset.country}/plots/comparisons/"
    os.makedirs(folder_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{target_name}.png")

def plot_comparisons_two_targets(names, targets, config_path, target_name=None, sub_target_name=None):
    if target_name is None:
        target_name = targets[0][-1]
        sub_target_name = targets[1][-1]

    sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    palette = sns.cubehelix_palette(n_colors=len(names), start=2.6, rot=0.1, dark=0.2, light=0.8)

    plt.suptitle(f"Comparison of performances of models based on {target_name} and {sub_target_name}")
    results = []
    for name in names:
        cfg = load_hydra_config(name, config_path)
        target = cfg[targets[0][0]][targets[0][1]]
        sub_target = cfg[targets[1][0]].get(targets[1][1], 0.0)

        with open(f"../logs/{cfg.dataset.country}/{name}.json", "r") as f:
            ndcg = json.load(f)["valid"]["ndcg"]

        df = pd.DataFrame({
            "epoch": range(len(ndcg)),
            "ndcg": ndcg,
            "main_target": [target] * len(ndcg),
            "sub_target": [sub_target] * len(ndcg),
            "target": [f"{target} / {sub_target}"] * len(ndcg)
        })

        results.append(df)

    results = pd.concat(results, axis=0).reset_index(drop=True)

    plot_history_line(results, ax1, palette)
    plot_ndcg_bar(results, ax2, palette[::results["main_target"].nunique()], target_name, sub_target_name)

    folder_path = f"../results/{cfg.dataset.country}/plots/comparisons/"
    os.makedirs(folder_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{target_name}_{sub_target_name}.png")

if __name__ == "__main__":
    config_path = "../config"

    names = ["emb_32", "baseline", "emb_128", "emb_256", "emb_512", "emb_1024"]
    targets = ["model", "embedding_dim"]
    plot_comparisons_one_target(names, targets, config_path)

    names = ["n_layers_2", "baseline", "n_layers_4", "n_layers_5", "n_layers_6", "n_layers_7", "n_layers_8"]
    targets = ["model", "n_layers"]
    plot_comparisons_one_target(names, targets, config_path)

    names = ["reg_0.01", "reg_0.001", "baseline", "reg_0.00001"]
    targets = ["training", "reg_weight"]
    plot_comparisons_one_target(names, targets, config_path)

    names = ["baseline", "batch_2048", "batch_4096"]
    targets = ["training", "batch_size"]
    plot_comparisons_one_target(names, targets, config_path)

    names = ["lr_0.01", "baseline", "lr_0.0001"]
    targets = ["training", "lr"]
    plot_comparisons_one_target(names, targets, config_path)

    names = [
        "emb_512_n_layers_4_reg_0.001",
        "emb_512_n_layers_5_reg_0.001",
        "emb_512_n_layers_6_reg_0.001",
        "emb_512_n_layers_7_reg_0.001"
    ]
    targets = ["model", "n_layers"]
    plot_comparisons_one_target(names, targets, config_path, target_name="emb_512_reg_0.001_n_layers")


    names = [
        "emb_512_n_layers_5_reg_0.001",
        "emb_512_n_layers_5_reg_0.001_gentle_0.1",
        "emb_512_n_layers_5_reg_0.001_gentle_0.05",
        "emb_512_n_layers_6_reg_0.001",
        "emb_512_n_layers_6_reg_0.001_gentle_0.1",
        "emb_512_n_layers_6_reg_0.001_gentle_0.05",
        "emb_512_n_layers_7_reg_0.001",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1",
        "emb_512_n_layers_7_reg_0.001_gentle_0.05",
        "emb_512_n_layers_8_reg_0.001",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1",
        "emb_512_n_layers_8_reg_0.001_gentle_0.05",
    ]

    targets = [["model", "n_layers"], ["model", "layer_decay"]]
    plot_comparisons_two_targets(names, targets, config_path, target_name=None, sub_target_name=None)

    names = [
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl"
    ]
    targets = ["model", "n_layers"]
    plot_comparisons_one_target(names, targets, config_path, target_name="emb_512_reg_0.001_gentle_0.1_gcl_n_layers")

    names = [
        "emb_512_n_layers_6_reg_0.001_gentle_0.1",
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl",
    ]
    targets = [["model", "n_layers"], ["training", "gcl_reg"]]
    plot_comparisons_two_targets(names, targets, config_path, target_name=None, sub_target_name=None)

    names = [
        "time_decay_0.001_0.1",
        "time_decay_0.001_0.3",
        "time_decay_0.0005_0.1",
        "time_decay_0.0005_0.3",
        "time_decay_0.0001_0.1",
        "time_decay_0.0001_0.3",
    ]
    targets = [["model", "time_decay"], ["model", "time_min_weight"]]
    plot_comparisons_two_targets(names, targets, config_path, target_name=None, sub_target_name=None)

    names = [
        "baseline",
        "geo_distance_20.0_1.0",
        "geo_distance_20.0_1.5",
        "geo_distance_20.0_2.0",
        "geo_distance_20.0_3.0",
        "geo_distance_20.0_4.0",
        "geo_distance_20.0_5.0",
        "geo_distance_20.0_6.0",
        "geo_distance_20.0_7.0",
        "geo_distance_20.0_8.0",
    ]
    targets = ["model", "geo_sigma"]
    plot_comparisons_one_target(names, targets, config_path, target_name=None)

    names = [
        "baseline",
        "geo_distance_5",
        "geo_distance_10",
        "geo_distance_15",
        "geo_distance_20",
    ]
    targets = ["model", "geo_k_neighbors"]
    plot_comparisons_one_target(names, targets, config_path, target_name=None)
