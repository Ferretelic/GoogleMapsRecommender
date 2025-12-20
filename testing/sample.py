import pandas as pd

def sample_test_users(cfg):
     train_df = pd.read_csv(f"{self.cfg.paths.splits}/train.csv")
    valid_df = pd.read_csv(f"{self.cfg.paths.splits}/valid.csv")
    test_df = pd.read_csv(f"{self.cfg.paths.splits}/test.csv")