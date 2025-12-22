import os

import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

def plot_num_places_per_region(meta, region, folder_path, category):
    meta = meta[region].value_counts().reset_index()
    meta = meta.sort_values(by="count")[::-1]

    if meta.shape[0] > 51:
        meta = meta[:50]

    plt.figure(figsize=(20, 15))
    sns.barplot(meta, x="count", y=region, hue=region)

    plt.title(f"The number of {category} places for each {region}")
    plt.xlabel(f"The number of {category} places")
    plt.ylabel(region)

    plt.tight_layout()
    plt.savefig(f"{folder_path}/num_places_per_region.png")
    plt.close()

def plot_num_places_per_user(review, region, folder_path, category):
    review = review[["user_id", region]].groupby("user_id").agg(["count", "nunique"]).reset_index()
    review.columns = ["user_id", "count", "nunique"]

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 15))

    sns.histplot(review, x="count", ax=axes[0], bins=np.arange(5, 50), alpha=0.8)
    axes[0].set_title(f"The distribution of the number of reviews per user.")
    axes[0].set_xlabel(f"The number of reviews per user.")
    axes[0].set_ylabel(f"Frequency")

    sns.histplot(review, x="nunique", ax=axes[1], bins=np.arange(30), alpha=0.8)
    axes[1].set_title(f"The distribution of the number of {region} per user.")
    axes[1].set_xlabel(f"The number of {region} per user.")
    axes[1].set_ylabel(f"Frequency")

    plt.tight_layout()
    plt.savefig(f"{folder_path}/num_places_per_user.png")
    plt.close()

def plot_name_distribution(meta, folder_path, category):
    meta = meta["name"].value_counts().reset_index()[:50]

    plt.figure(figsize=(20, 15))
    sns.barplot(meta, x="count", y="name", hue="name")

    plt.title(f"The number of {category} places with shared name.")
    plt.xlabel(f"The number of {category} places,")
    plt.ylabel("The name of the place.")

    plt.tight_layout()
    plt.savefig(f"{folder_path}/name_distribution.png")
    plt.close()

def visualize_dataset_features(cfg):
    sns.set_theme(style="whitegrid", palette="hls")

    folder_path = f"{cfg.paths.result}/plots/visualization"
    os.makedirs(folder_path, exist_ok=True)
    category = cfg.dataset.category

    meta = load_combined_datast(cfg, "meta")
    review = load_combined_datast(cfg, "review")

    if len(cfg.dataset.get("states", [])) == 1:
        pattern = r",\s+([^,]+),\s+CA"
        meta["city"] = meta["address"].str.extract(pattern)
        gmap2city = {gmap_id: city for gmap_id, city in meta[["gmap_id", "city"]].values}
        review["city"] = review["gmap_id"].apply(lambda x: gmap2city[x])

        plot_num_places_per_region(meta, "city", folder_path, category)
        plot_num_places_per_user(review, "city", folder_path, category)
        plot_name_distribution(meta, folder_path, category)

    else:
        plot_num_places_per_region(meta, "state", folder_path, category)
        plot_num_places_per_user(review, "state", folder_path, category)
        plot_name_distribution(meta, folder_path, category)
