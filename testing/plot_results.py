import math

import matplotlib.pyplot as plt
import seaborn as sns

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

    # 淡いブルーと淡いパープル
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

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')

    fig.suptitle("Recommender System Evaluation Dashboard", fontsize=24, weight="bold", y=0.98)

    add_radar_chart(df, fig, cmap)
    add_scatter_plot(df, fig, cmap)
    add_bar_plot(df, fig)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{cfg.paths.result}/test_results.png")