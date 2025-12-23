import os

import torch
from sklearn.decomposition import PCA
import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

def apply_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embeddings)

    df = pd.DataFrame(components, columns=[f"component_{i}" for i in range(n_components)])
    return df

def apply_umap(embeddings, n_components, targets):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=40)
    components = reducer.fit_transform(embeddings, y=targets)

    df = pd.DataFrame(components, columns=[f"component_{i}" for i in range(n_components)])
    return df

def get_gmap2category_mappings(cfg, gmap_ids):
    meta = load_combined_datast(cfg, "meta")
    targets = {"gmap_id": gmap_ids}
    for category in cfg.analysis.item_category:
        if category == "state":
            top_states = meta["state"].value_counts().nlargest(10).index.tolist()
            meta["state"] = meta["state"].where(meta["state"].isin(top_states), "Other")

            numerical_mapping = {city: i for i, city in enumerate(top_states)}
            numerical_mapping["Other"] = None
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
            for gmap_id, name in meta[["gmap_id", "name"]].values:
                if cfg.dataset.category == "Cafe":
                    if "starbucks" in name.lower():
                        gmap2category[gmap_id] = ("Starbucks", 1)
                    elif "mcdonald" in name.lower():
                        gmap2category[gmap_id] = ("McDonald's", 2)
                    else:
                        gmap2category[gmap_id] = ("Other", 0)
                elif cfg.dataset.category == "Asian":
                    if "panda express" in name.lower():
                        gmap2category[gmap_id] = ("Panda Express", 1)
                    else:
                        gmap2category[gmap_id] = ("Other", 0)
                else:
                    gmap2category[gmap_id] = ("Other", 0)

        elif category == "popularity":
            for gmap_id, num_reviews in meta[["gmap_id", "num_of_reviews"]].values:
                if num_reviews <= 100:
                    gmap2category[gmap_id] = ("unpopular", 0)
                elif num_reviews >= 500:
                    gmap2category[gmap_id] = ("popular", 2)
                else:
                    gmap2category[gmap_id] = ("neutral", 1)

        elif category == "price":
            meta["price"] = meta["price"].fillna("")
            for gmap_id, price in meta[["gmap_id", "price"]].values:
                if price is None:
                    gmap2category[gmap_id] = ("None", 0)
                elif "$" in price:
                    gmap2category[gmap_id] = (r"\$" * len(price), len(price))
                else:
                    gmap2category[gmap_id] = ("None", 0)

        elif category == "city":
            pattern = r",\s+([^,]+),\s+CA"
            meta["city"] = meta["address"].str.extract(pattern)
            top_cities = meta["city"].value_counts().nlargest(10).index.tolist()
            meta["city"] = meta["city"].where(meta["city"].isin(top_cities), "Other")

            numerical_mapping = {city: i for i, city in enumerate(top_cities)}
            numerical_mapping["Other"] = None

            gmap2category = {gmap_id: (c, numerical_mapping[c]) for (gmap_id, c) in meta[["gmap_id", "city"]].values}

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
        _, axes = plt.subplots(2, 3, figsize=(40, 20))
        pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    axes = axes.flatten()

    plt.suptitle(f"Distribution of dimensionaly reduced item embeddings with {method} colored by {category}")
    for index in range(len(pairs)):
        i, j = pairs[index]
        ax = axes[index]

        sns.scatterplot(df, x=f"component_{i}", y=f"component_{j}", ax=ax, hue=category)

        ax.set_xlabel(f"component {i+1}")
        ax.set_ylabel(f"component {j+1}")

    plt.tight_layout()
    folder_path = f"{cfg.paths.result}/plots/dim_reduction/{model}/{method}"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f"{folder_path}/{category}.png")
    plt.close()

def mask_out_embeddings(item_embs, df, target_col):
    mask = df[target_col].notnull().values
    masked_embs = item_embs[mask]

    norm = np.linalg.norm(masked_embs, axis=1, keepdims=True)
    masked_embs = masked_embs / (norm + 1e-10)

    category_df = df[mask].copy()
    category_df = category_df.reset_index(drop=True)
    category_df[target_col] = category_df[target_col].astype("int32")

    return masked_embs, category_df

def analyze_item_embeddings(cfg, item_embs, index2gmap, model):
    print(f"  Start analysing embedding of {model}")
    n_components = cfg.analysis.n_components

    gmap_ids = [index2gmap[str(index)] for index in range(item_embs.shape[0])]
    targets = get_gmap2category_mappings(cfg, gmap_ids)
    df = pd.DataFrame(targets)

    for method in cfg.analysis.methods:
        for category in cfg.analysis.item_category:
            file_path = f"{cfg.paths.result}/plots/dim_reduction/{model}/{method}/{category}.png"
            if os.path.exists(file_path):
                continue

            if category == "state" and len(cfg.dataset.get("states", [])) == 1:
                continue
            if category == "chain" and cfg.dataset.category not in ["Asian", "Cafe"]:
                continue

            target_col = f"{category}_numerical"
            masked_embs, category_df = mask_out_embeddings(item_embs, df, target_col)

            if method == "pca":
                print(f"    Applying PCA to embeddings from {model} with {category}")
                components = apply_pca(masked_embs, n_components)
            elif method == "umap":
                print(f"    Applying UMAP to embeddings from {model} with {category}")
                components = apply_umap(masked_embs, n_components, category_df[target_col])
            else:
                raise NotImplementedError

            category_df = pd.concat([category_df, components], axis=1)
            plot_item_distribution(cfg, category_df, model, method, category)

def analyze_embeddings(cfg, model):
    model_path = f"{cfg.paths.embedding}/{model}.pt"
    user_embs, item_embs = torch.load(model_path, map_location="cpu")

    user_embs = user_embs.detach().numpy()
    item_embs = item_embs.detach().numpy()

    mappings = load_mappings(cfg)
    index2gmap = mappings["index2gmap"]

    analyze_item_embeddings(cfg, item_embs, index2gmap, model)