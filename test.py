import hydra
from omegaconf import DictConfig
import pandas as pd

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

if __name__ == "__main__":
    main()