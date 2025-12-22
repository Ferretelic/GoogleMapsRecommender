from functools import partial

import hydra
from omegaconf import DictConfig

from training.lightgcn import *
from training.trainer import *

from utils import *

def train_model(cfg):
    if check_training_started(cfg):
        return

    print("    Constructing LightGCN...")
    model = LightGCN(cfg)

    print("    Training LightGCN...")
    trainer = Trainer(cfg, model)
    trainer.train()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    seed_everything()

    print("Training of Model")
    train_model(cfg)

if __name__ == "__main__":
    main()