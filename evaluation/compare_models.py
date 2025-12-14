import os
import json

import numpy as np
import pandas as pd

def update_performance_csv(cfg):
    metrics = []
    for log_file in os.listdir(f"{cfg.paths.logs}"):
        with open(f"{cfg.paths.logs}/{log_file}", "r") as f:
            results = json.load(f)

        recall = results["valid"]["recall"]
        ndcg = results["valid"]["ndcg"]

        min_index = np.argmax(ndcg)
        metrics.append((log_file.replace(".log", ""), recall[min_index], ndcg[min_index], min_index+1))

    df = pd.DataFrame(metrics, columns=["name", "recall", "ndcg", "epoch"])
    df = df.sort_values(by="ndcg").reset_index(drop=True)

    print(df)

    df.to_csv(f"{cfg.paths.result}/performances.csv", index=False)
