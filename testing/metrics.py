import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

def plot_training_history(cfg):
    for log_file in os.listdir(cfg.paths.logs):
        model = log_file.replace(".json", "")
        with open(f"{cfg.paths.logs}/{log_file}", "r") as f:
            logs = json.load(f)

        train_loss = logs["train"]
        valid_recall = logs["valid"]["recall"]
        valid_ndcg = logs["valid"]["ndcg"]

        epochs = range(1, len(train_loss) + 1)

        sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.lineplot(x=epochs, y=train_loss, ax=ax1, color="#2c3e50", linewidth=2.5, label="Train Loss")
        ax1.set_title("Training Loss Curve", fontsize=16, fontweight="bold", pad=15)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

        sns.lineplot(x=epochs, y=valid_recall, ax=ax2,
                    color="#27ae60", linewidth=2.5, label="Recall")
        sns.lineplot(x=epochs, y=valid_ndcg, ax=ax2,
                    color="#e67e22", linewidth=2.5, label="NDCG")

        ax2.set_title("Validation Metrics (Recall & NDCG)", fontsize=16, fontweight="bold", pad=15)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Score", fontsize=12)
        ax2.legend(loc="lower right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

        os.makedirs(f"{cfg.paths.result}/history", exist_ok=True)
        plt.savefig(f"{cfg.paths.result}/history/{model}.png")

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

def plot_history_line(results, metric, ax, palette):
    sns.lineplot(results, x="epoch", y=metric, ax=ax, hue="name", palette=palette)

    ax.set_title(f"Comparison of training history of {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()

def plot_ndcg_bar(results, metric, ax, palette):
    results = results[results["best"]]
    sns.barplot(results, x="name", y=metric, ax=ax, hue="name", palette=palette)

    ax.set_title(f"Comparison of valid {metric}")
    ax.set_xlabel("Models")
    ax.set_ylabel(metric)

def plot_validation_results(cfg):
    results = []
    for log_file in os.listdir(cfg.paths.logs):
        name = log_file.replace(".json", "")

        with open(f"{cfg.paths.logs}/{log_file}", "r") as f:
            valid = json.load(f)["valid"]

        recall, ndcg = valid["recall"], valid["ndcg"]

        best_index = np.argmax(ndcg)
        best = [False] * len(ndcg)
        best[best_index] = True
        df = pd.DataFrame({
            "name": [name] * len(ndcg),
            "epoch": range(len(ndcg)),
            "best": best,
            "recall": recall,
            "ndcg": ndcg,
        })

        results.append(df)

    results = pd.concat(results, axis=0).reset_index(drop=True)

    sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})
    palette = "Blues"
    for metric in ["recall", "ndcg"]:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        plt.suptitle(f"Comparison of performances of models on validation dataset")

        plot_history_line(results, metric, ax1, palette)
        plot_ndcg_bar(results, metric, ax2, palette)

        folder_path = f"{cfg.paths.result}/plots/comparisons/"
        os.makedirs(folder_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{folder_path}/{metric}.png")
        plt.close()

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