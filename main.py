from functools import partial

import hydra
from omegaconf import DictConfig

from processing.download_file import *
from processing.process_raw_meta import *
from processing.process_raw_review import *
from processing.combine_file import *
from processing.split_dataset import *

from training.lightgcn import *
from training.trainer import *

def process_dataset(cfg):
    if not check_process_complete(cfg):
        print("    Downloading dataset...")
        download_dataset(cfg)

        print("    Processing raw meta data...")
        country_filter = partial(
            filter_by_country, country=cfg.dataset.country)

        num_reviews_filter = partial(
            filter_by_num_reviews, min_num_reviews=cfg.dataset.min_num_reviews)

        filter_raw_meta_data(cfg, [country_filter, num_reviews_filter])

        print("    Processing raw review data...")
        filter_raw_review_data(cfg)

    print("    Combining state data files...")
    combine_state_files(cfg, "meta")
    combine_state_files(cfg, "review")

    print("    Splitting dataset into train, valid, and test...")
    split_dataset_by_temporal(cfg)

def train_model(cfg):
    print("    Constructing LightGCN...")
    model = LightGCN(cfg)

    print("    Training LightGCN...")
    trainer = Trainer(cfg, model)
    trainer.train()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Preparation of Dataset")
    process_dataset(cfg)

    print("Training of Model")
    train_model(cfg)

if __name__ == "__main__":
    main()