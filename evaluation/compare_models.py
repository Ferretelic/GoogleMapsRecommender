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
    log_folder = f"./logs/{cfg.dataset.category}"

    metrics = []
    for name in os.listdir(log_folder):
        cfg = load_hydra_config(name.replace(".json", ""), config_path)

        with open(f"./logs/{cfg.dataset.category}/{name}", "r") as f:
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

    df.to_csv(f"./results/{cfg.dataset.category}/performances.csv", index=False)

def load_hydra_config(name, config_path):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=name, overrides=[])
        return cfg

if __name__ == "__main__":
    config_path = "../config"
    update_performances(config_path, view_columns=view_columns)