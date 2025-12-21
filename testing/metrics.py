import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

def update_valid_performances(cfg):
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

    metrics = []
    for name in os.listdir(cfg.paths.logs):
        model_cfg = load_hydra_config(name.replace(".json", ""))

        with open(f"{cfg.paths.logs}/{name}", "r") as f:
            results = json.load(f)["valid"]

        n_patience = len(results["ndcg"]) - (np.argmax(results["ndcg"]) + 1)
        best_index = np.argmax(results["ndcg"])
        recall, ndcg = results["recall"][best_index], results["ndcg"][best_index]
        metric = [recall, ndcg, n_patience]

        metric += [model_cfg[target[0]].get(target[1], default) for (target, default) in hyperparameters]
        metric += [name.replace(".json", "")]
        metrics.append(metric)

    columns = ["recall", "ndcg", "n_patience"] + [target[1] for (target, _) in hyperparameters] + ["name"]
    df = pd.DataFrame(metrics, columns=columns).sort_values(by="ndcg").reset_index(drop=True)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    print(df[["name", "recall", "ndcg", "n_patience"]])

    df.to_csv(f"{cfg.paths.result}/valid_performances.csv", index=False)

def add_radar_chart(df, fig, cmap):
    ax1 = fig.add_subplot(221, polar=True)

    metrics = ["recall", "ndcg", "diversity", "coverage"]
    normalized_df = df.copy()
    for col in metrics:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val == 0:
            normalized_df[col] = 0
        else:
            normalized_df[col] = (df[col] - min_val) / (max_val - min_val)

    labels = [m.upper() for m in metrics]
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    target_models = df["name"].unique()

    for model_name in target_models:
        row = normalized_df[normalized_df["name"] == model_name]
        values = row[metrics].values.flatten().tolist()
        values += values[:1]

        ax1.plot(angles, values, linewidth=2, linestyle="solid", label=model_name, color=cmap[model_name])
        ax1.fill(angles, values, color=cmap[model_name], alpha=0.1)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=12, weight="bold")

    ax1.spines["polar"].set_visible(False)
    ax1.set_title("Model Capability Profile (Normalized)", size=15, pad=10)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=9)

def add_scatter_plot(df, fig, cmap):
    ax2 = fig.add_subplot(222)

    sns.scatterplot(
        data=df, x="diversity", y="recall",
        hue="name", palette=cmap, s=300, alpha=0.9, ax=ax2,
        edgecolor="white"
    )

    best_model_name = "LightGCN Best (Dot)"
    if best_model_name in df["name"].values:
        best_model = df[df["name"] == best_model_name]
        ax2.scatter(
            best_model["diversity"], best_model["recall"],
            s=600, facecolors="none", edgecolors="#666666", linewidth=2, linestyle="--"
        )

    ax2.set_title("The Trade-off Frontier: Accuracy vs Diversity", size=15)
    ax2.set_xlabel("Diversity", fontsize=12)
    ax2.set_ylabel("Recall", fontsize=12)

    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend().remove()

def add_bar_plot(df, fig):
    ax3 = fig.add_subplot(212)

    prr_df = df[["name", "head_prr", "tail_prr"]].melt(id_vars="name", var_name="Type", value_name="PRR")

    bar_palette = {"head_prr": "#a2d2ff", "tail_prr": "#cdb4db"}

    sns.barplot(
        data=prr_df, x="name", y="PRR", hue="Type",
        palette=bar_palette, ax=ax3, alpha=0.9,
        edgecolor="white"
    )

    ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Ideal Calibration (1.0)")

    ax3.set_title("Popularity Calibration (Closer to 1.0 is Better)", size=15)
    ax3.set_ylabel("Popularity Recommendation Ratio (PRR)", fontsize=12)
    ax3.set_xlabel("")

    ax3.legend(loc="upper left")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

def plot_test_results(cfg, df):
    colors = sns.color_palette("pastel", len(df))
    cmap = {name: col for name, col in zip(df["name"], colors)}

    fig = plt.figure(figsize=(25, 12))
    fig.patch.set_facecolor('white')

    fig.suptitle("Recommender System Evaluation Dashboard", fontsize=24, weight="bold", y=0.98)

    add_radar_chart(df, fig, cmap)
    add_scatter_plot(df, fig, cmap)
    add_bar_plot(df, fig)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{cfg.paths.result}/test_performances.png")