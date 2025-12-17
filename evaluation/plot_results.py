import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(cfg):
    with open(f"{cfg.paths.logs}/{cfg.name}.json", "r") as f:
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

    os.makedirs(f"{cfg.paths.plots}", exist_ok=True)
    plt.savefig(f"{cfg.paths.plots}/history.png")