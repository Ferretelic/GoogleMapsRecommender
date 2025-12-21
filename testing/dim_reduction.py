
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

    analyze_item_embeddings(cfg, item_embs, index2gmap)

def apply_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embeddings)

    df = pd.DataFrame(components, columns=[f"component_{i}" for i in range(n_components)])
    return df

def apply_umap(embeddings, n_components):
    reducer = umap.UMAP(n_components=n_components)
    components = reducer.fit_transform(embeddings)

    df = pd.DataFrame(components, columns=[f"component_{i}" for i in range(n_components)])
    return df

def add_category_data(df, item_category):
    meta = load_combined_datast("meta")

    for category in item_category:
        gmap2category = {gmap_id: c for (gmap_id, c) in meta[["gmap_id", category]].values}
        df[category] = df["gmap_id"].apply(lambda x: gmap2category[x])

    return df

def plot_item_distribution(df, n_components, category):
    sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})

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

    for index in range(n_components):
        i, j = pairs[index]
        sns.scatterplot(df, x=f"component_{i}", y=f"component_{j}", ax=axes[index])


def analyze_item_embeddings(cfg, item_embs, index2gmap):
    n_components = cfg.analysis.n_components

    for method in cfg.analysis.methods:
        if method == "pca":
            df = apply_pca(item_embs, n_components)
        elif method == "umap":
            df = apply_umap(item_embs, n_components)
        else:
            raise NotImplementedError

        gmap_ids = [index2gmap[str(index)] for index in range(item_embs.shape[0])]
        df["gmap_id"] = gmap_ids
