import os
import json

import numpy as np
import pandas as pd

def construct_df():
    metrics = []
    for log_file in os.listdir("../logs/Japanese"):
        with open(f"../logs/Japanese/{log_file}", "r") as f:
            results = json.load(f)

        recall = results["valid"]["recall"]
        ndcg = results["valid"]["ndcg"]

        min_index = np.argmax(ndcg)
        metrics.append((log_file.replace(".log", ""), recall[min_index], ndcg[min_index]))

    df = pd.DataFrame(metrics, columns=["name", "recall", "ndcg"])
    df = df.sort_values(by="ndcg")

    print(df)

if __name__ == "__main__":
    construct_df()
