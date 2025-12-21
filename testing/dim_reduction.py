import os

import torch
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

def analyze_embeddings(cfg, model):
    model_path = f"{cfg.paths.embedding}/{model}.pt"
    user_embs, item_embs = torch.load(model_path, map_location="cpu")

    user_embs = user_embs.detach().numpy()
    item_embs = item_embs.detach().numpy()

    mappings = load_mappings(cfg)
    index2gmap = mappings["index2gmap"]

    analyze_item_embeddings(cfg, item_embs, index2gmap, model)

def apply_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embeddings)

    df = pd.DataFrame(components, columns=[f"component_{i}" for i in range(n_components)])
    return df

def apply_umap(embeddings, n_components, targets):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=100)
    components = reducer.fit_transform(embeddings, y=targets)

    df = pd.DataFrame(components, columns=[f"component_{i}" for i in range(n_components)])
    return df

def get_gmap2category_mappings(cfg, gmap_ids):
    meta = load_combined_datast(cfg, "meta")

    targets = {"gmap_id": gmap_ids}
    for category in cfg.analysis.item_category:
        if category == "state":
            numerical_mapping = {c:i for i, c in enumerate(np.sort(meta[category].unique()))}
            gmap2category = {gmap_id: (c, numerical_mapping[c]) for (gmap_id, c) in meta[["gmap_id", category]].values}

        elif category == "rating":
            gmap2category = {}
            for gmap_id, avg_rating in meta[["gmap_id", "avg_rating"]].values:
                if avg_rating < 3.5:
                    gmap2category[gmap_id] = ("low", 0)
                elif avg_rating > 4.5:
                    gmap2category[gmap_id] = ("high", 2)
                else:
                    gmap2category[gmap_id] = ("neutral", 1)

        elif category == "chain":
            gmap2category = {}
            print(meta.columns)
            for gmap_id, name in meta[["gmap_id", "name"]].values:
                if "starbucks" in name.lower():
                    gmap2category[gmap_id] = ("Starbucks", 1)
                elif "mcdonald" in name.lower():
                    gmap2category[gmap_id] = ("McDonald's", 2)
                else:
                    gmap2category[gmap_id] = ("Other", 0)

        elif category == "popularity":
            for gmap_id, num_reviews in meta[["gmap_id", "num_of_reviews"]].values:
                if num_reviews <= 10:
                    gmap2category[gmap_id] = ("unpopular", 0)
                elif num_reviews >= 100:
                    gmap2category[gmap_id] = ("popular", 2)
                else:
                    gmap2category[gmap_id] = ("neutral", 1)

        else:
            raise NotImplementedError

        categories = [gmap2category[gmap_id][0] for gmap_id in gmap_ids]
        numericals = [gmap2category[gmap_id][1] for gmap_id in gmap_ids]
        targets[category] = categories
        targets[f"{category}_numerical"] = numericals

    return targets

def plot_item_distribution(cfg, df, model, method, category):
    n_components = cfg.analysis.n_components
    sns.set_theme(style="whitegrid", palette="hls")

    if n_components == 2:
        _, axes = plt.subplots(1, figsize=(15, 15))
        pairs = [[0, 1]]
    elif n_components == 3:
        _, axes = plt.subplots(1, 2, figsize=(20, 10))
        pairs = [[0, 1], [0, 2], [1, 2]]
    elif n_components == 4:
        _, axes = plt.subplots(2, 3, figsize=(25, 15))
        pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    axes = axes.flatten()

    plt.suptitle(f"Distribution of dimensionaly reduced item embeddings with {method} colored by {category}")
    for index in range(len(pairs)):
        i, j = pairs[index]
        ax = axes[index]

        sns.scatterplot(df, x=f"component_{i}", y=f"component_{j}", ax=ax, hue=category)

        ax.set_xlabel(f"{i+1}-th component")
        ax.set_ylabel(f"{j+1}-th component")

    plt.tight_layout()
    folder_path = f"{cfg.paths.result}/dim_reduction/{model}"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f"{folder_path}/{method}_{category}.png")

def analyze_item_embeddings(cfg, item_embs, index2gmap, model):
    n_components = cfg.analysis.n_components

    gmap_ids = [index2gmap[str(index)] for index in range(item_embs.shape[0])]
    targets = get_gmap2category_mappings(cfg, gmap_ids)
    df = pd.DataFrame(targets)

    for method in cfg.analysis.methods:
        for category in cfg.analysis.item_category:
            if method == "pca":
                print(f"  Applying PCA to embeddings from {model} with {category}")
                components = apply_pca(item_embs, n_components)
            elif method == "umap":
                print(f"  Applying UMAP to embeddings from {model} with {category}")
                components = apply_umap(item_embs, n_components, df[f"{category}_numerical"])
            else:
                raise NotImplementedError

            category_df = pd.concat([df, components], axis=1)
            plot_item_distribution(cfg, category_df, model, method, category)
