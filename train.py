from functools import partial

import hydra
from omegaconf import DictConfig

from training.lightgcn import *
from training.trainer import *
from training.evaluation import *

from utils import *

def train_model(cfg):
    if check_training_started(cfg):
        return

    print("    Constructing LightGCN...")
    model = LightGCN(cfg)

    print("    Training LightGCN...")
    trainer = Trainer(cfg, model)
    trainer.train()

def evaluate_model(cfg):
    if not check_training_completed(cfg):
        return

    print("    Plotting training history...")
    plot_training_history(cfg)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    seed_everything()

    print("Training of Model")
    train_model(cfg)

    print("Evaluation of Model")
    evaluate_model(cfg)

if __name__ == "__main__":
    main()