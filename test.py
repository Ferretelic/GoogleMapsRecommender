import math

import hydra
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from testing.tester import *

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig):
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

def add_radar_chart(df, fig, cmap):
    ax1 = fig.add_subplot(221, polar=True)

    metrics = ["recall", "ndcg", "diversity", "coverage"]
    normalized_df = df.copy()
    for col in metrics:
        min_val = df[col].min()
        max_val = df[col].max()
        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)

    labels = [m.upper() for m in metrics]
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    target_models = ["KNN", "Popularity", "LightGCN Baseline (Dot)", "LightGCN Best (Dot)", "LightGCN Best (Cosine)"]
    for model_name in target_models:
        row = normalized_df[normalized_df["name"].lower() == model_name.lower()]
        values = row[metrics].values.flatten().tolist()
        values += values[:1]

        ax1.plot(angles, values, linewidth=2, linestyle="solid", label=model_name, color=cmap[model_name.lower()])
        ax1.fill(angles, values, color=cmap[model_name.lower()], alpha=0.1)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=12, color="white", weight="bold")
    ax1.set_title("Model Capability Profile (Normalized)", size=15, color="cyan", pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax1.grid(color="#444444")
    ax1.spines["polar"].set_visible(False)

def add_scatter_plot(df, fig, cmap):
    ax2 = fig.add_subplot(222)

    sns.scatterplot(
        data=df, x="diversity", y="recall",
        hue="name", palette=cmap, s=300, alpha=0.9, ax=ax2, edgecolor="white"
    )

    best_model = df[df["name"] == "LGCN Best (Dot)"]
    ax2.scatter(best_model["diversity"], best_model["recall"], s=600, facecolors="none", edgecolors="yellow", linewidth=2, linestyle="--")
    ax2.text(best_model["diversity"].values[0]+0.02, best_model["recall"].values[0], " Champion!", color="yellow", fontsize=12, va="center")

    ax2.set_title("The Trade-off Frontier: Accuracy vs Diversity", size=15, color="cyan")
    ax2.set_xlabel("Diversity", fontsize=12)
    ax2.set_ylabel("Recall", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend().remove()

def add_bar_plot(df, fig):
    ax3 = fig.add_subplot(212)

    prr_df = df[["name", "head_prr", "tail_prr"]].melt(id_vars="name", var_name="Type", value_name="PRR")

    bar_plot = sns.barplot(
        data=prr_df, x="name", y="PRR", hue="Type",
        palette={"head_prr": "#ff0055", "tail_prr": "#00ffaa"}, ax=ax3, alpha=0.8
    )

    ax3.axhline(1.0, color="yellow", linestyle="--", linewidth=2, label="Ideal Calibration (1.0)")

    idx = df[df["name"] == "LGCN Best (Dot)"].index[0]
    ax3.annotate("Perfect Balance!",
                xy=(idx, 1.1), xytext=(idx, 1.5),
                arrowprops=dict(facecolor="white", shrink=0.05),
                fontsize=12, color="white", ha="center", weight="bold")

    ax3.set_title("Popularity Calibration (Closer to 1.0 is Better)", size=15, color="cyan")
    ax3.set_ylabel("Popularity Recommendation Ratio (PRR)", fontsize=12)
    ax3.set_xlabel("")
    ax3.legend(loc="upper left")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

def plot_test_results(cfg, df):
    plt.style.use("dark_background")
    colors = sns.color_palette("husl", len(df))
    cmap = {name: col for name, col in zip(df["name"], colors)}

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Recommender System Evaluation Dashboard", fontsize=24, color="white", weight="bold", y=0.95)

    add_radar_chart(df, fig, cmap)
    add_scatter_plot(df, fig, cmap)
    add_bar_plot(df, fig)

    plt.tight_layout()
    plt.savefig(f"{cfg.paths.result}/test_results.png")

if __name__ == "__main__":
    main()