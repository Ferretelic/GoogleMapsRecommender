import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hydra import initialize, compose

def update_performances(config_path, view_columns=None):
    hyperparameters = [
        (["training", "batch_size"], 1024),
        (["training", "lr"], 0.001),
        (["training", "reg_weight"], 0.0001),

        (["training", "gcl_reg"], 0.1),
        (["training", "gcl_eps"], 0.1),
        (["training", "gcl_temp"], 0.3),

        (["model", "embedding_dim"], 64),
        (["model", "n_layers"], 3),

        (["model", "geo_threshold"], 0.0),
        (["model", "geo_k_neighbors"], 0),
        (["model", "geo_weight"], 1.0),

        (["model", "time_decay"], 0.0),
        (["model", "time_min_weight"], 0.0),

        (["model", "layer_weighting"], "uniform"),
        (["model", "layer_decay"], 0.0)
    ]

    cfg = load_hydra_config("baseline", config_path)
    log_folder = f"../logs/{cfg.dataset.country}"

    metrics = []
    for name in os.listdir(log_folder):
        cfg = load_hydra_config(name.replace(".json", ""), config_path)

        with open(f"../logs/{cfg.dataset.country}/{name}", "r") as f:
            results = json.load(f)["valid"]

        n_patience = len(results["ndcg"]) - (np.argmax(results["ndcg"]) + 1)
        best_index = np.argmax(results["ndcg"])
        recall, ndcg = results["recall"][best_index], results["ndcg"][best_index]
        metric = [recall, ndcg, n_patience]

        metric += [cfg[target[0]].get(target[1], default) for (target, default) in hyperparameters]
        metric += [name.replace(".json", "")]
        metrics.append(metric)

    columns = ["recall", "ndcg", "n_patience"] + [target[1] for (target, _) in hyperparameters] + ["name"]
    df = pd.DataFrame(metrics, columns=columns).sort_values(by="ndcg").reset_index(drop=True)

    if view_columns is None:
        view_columns = columns

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    print(df[["recall", "ndcg", "n_patience"] + view_columns + ["name"]])

    df.to_csv("../results/performances.csv", index=False)

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
        target = cfg[targets[0][0]].get(targets[0][1], 0.0)
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
    view_columns = ["geo_k_neighbors", "geo_weight"]
    update_performances(config_path, view_columns=view_columns)

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
    plot_comparisons_two_targets(names, targets, config_path)

    names = [
        "baseline",
        "gcl_temp_0.1",
        "gcl_temp_0.2",
        "gcl",
        "gcl_temp_0.4",
        "gcl_temp_0.5",
    ]
    targets = ["training", "gcl_temp"]
    plot_comparisons_one_target(names, targets, config_path)

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
    plot_comparisons_two_targets(names, targets, config_path)

    names = [
        "emb_512_n_layers_3_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_4_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_5_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl",
    ]
    targets = ["model", "n_layers"]
    plot_comparisons_one_target(names, targets, config_path, target_name="emb_512_reg_0.001_gentle_0.1_gcl_n_layers")

    names = [
        "emb_512_n_layers_3_reg_0.001_gcl",
        "emb_512_n_layers_3_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_4_reg_0.001_gcl",
        "emb_512_n_layers_4_reg_0.001_gentle_0.1_gcl"
    ]
    targets = [["model", "n_layers"], ["model", "layer_decay"]]
    plot_comparisons_two_targets(names, targets, config_path)

    names = [
        "time_decay_0.001_0.1",
        "time_decay_0.001_0.3",
        "time_decay_0.0005_0.1",
        "time_decay_0.0005_0.3",
        "time_decay_0.0001_0.1",
        "time_decay_0.0001_0.3",
    ]
    targets = [["model", "time_decay"], ["model", "time_min_weight"]]
    plot_comparisons_two_targets(names, targets, config_path)

    names = [
        "baseline",
        "geo_distance_neighbors_5_weight_1.0",
        "geo_distance_neighbors_10_weight_1.0",
        "geo_distance_neighbors_15_weight_1.0",
        "geo_distance_neighbors_20_weight_1.0",
    ]
    targets = ["model", "geo_k_neighbors"]
    plot_comparisons_one_target(names, targets, config_path)

    names = [
        "geo_distance_neighbors_5_weight_0.1",
        "geo_distance_neighbors_5_weight_1.0",
        "geo_distance_neighbors_10_weight_0.1",
        "geo_distance_neighbors_10_weight_1.0",
        "geo_distance_neighbors_15_weight_0.1",
        "geo_distance_neighbors_15_weight_1.0",
        "geo_distance_neighbors_20_weight_0.1",
        "geo_distance_neighbors_20_weight_1.0",
    ]
    targets = [["model", "geo_k_neighbors"], ["model", "geo_weight"]]
    plot_comparisons_two_targets(names, targets, config_path)

    names = [
        "baseline",
        "geo_distance_threshold_5.0_weight_0.1",
        "geo_distance_threshold_10.0_weight_0.1",
    ]
    targets = ["model", "geo_threshold"]
    plot_comparisons_one_target(names, targets, config_path)

    names = [
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl_geo_distance_15",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl_geo_distance_15",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl_geo_distance_15",
    ]
    targets = [["model", "n_layers"], ["model", "geo_k_neighbors"]]
    plot_comparisons_two_targets(names, targets, config_path)

    names = [
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_6_reg_0.001_gentle_0.1_gcl_geo_distance_20.0_8.0",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_7_reg_0.001_gentle_0.1_gcl_geo_distance_20.0_8.0",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl",
        "emb_512_n_layers_8_reg_0.001_gentle_0.1_gcl_geo_distance_20.0_8.0",
    ]
    targets = [["model", "n_layers"], ["model", "geo_threshold"]]
    plot_comparisons_two_targets(names, targets, config_path)


    names = [
        "emb_512_n_layers_4_reg_0.001_gcl",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_2_weight_1.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_3_weight_1.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_4_weight_1.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_5_weight_1.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_7_weight_1.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_10_weight_1.0"
    ]
    targets = ["model", "geo_k_neighbors"]
    plot_comparisons_one_target(names, targets, config_path, target_name="emb_512_n_layers_4_reg_0.001_gcl")

    names = [
        "emb_512_n_layers_4_reg_0.001_gcl",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_5_weight_1.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_5_weight_2.0",
        "emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_5_weight_5.0",
    ]
    targets = ["model", "geo_weight"]
    plot_comparisons_one_target(names, targets, config_path, target_name="emb_512_n_layers_4_reg_0.001_gcl_geo_distance_neighbors_5")