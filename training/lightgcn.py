import torch
import torch.nn as nn
import torch.nn.functional as F

from training.dataset import *

class LightGCN(nn.Module):
    def __init__(self, cfg):
        super(LightGCN, self).__init__()
        self.n_users, self.n_items = load_dataset_sizes(cfg)

        self.n_layers = cfg.model.n_layers
        self.embedding = nn.Embedding(self.n_users + self.n_items, cfg.model.embedding_dim)

        nn.init.xavier_normal_(self.embedding.weight)

        weight_method = cfg.model.get("weighting", "uniform")

        num_weights = self.n_layers + 1
        if weight_method == "uniform":
            weights = torch.ones(num_weights)
        elif weight_method == "gentle":
            decay = cfg.model.get("decay", 0.1)
            weights = torch.ones(num_weights) - torch.arange(num_weights) * decay
        else:
            raise NotImplementedError

        weights = weights / torch.sum(weights)
        self.register_buffer("layer_weights", weights)

    def forward(self, graph):
        all_emb = self.embedding.weight
        embs = [all_emb]

        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(graph, x)
            embs.append(x)

        embs = torch.stack(embs, dim=1)
        weights = self.layer_weights.view(1, -1, 1)
        light_out = torch.sum(embs * weights, dim=1)

        users_emb, items_emb = torch.split(light_out, [self.n_users, self.n_items])
        return users_emb, items_emb

    def load_embeddings(self, embeddings):
        users_emb, items_emb = embeddings

        device = self.embedding.weight.device
        users_emb = users_emb.to(device)
        items_emb = items_emb.to(device)

        new_weights = torch.cat([users_emb, items_emb], dim=0)
        with torch.no_grad():
            self.embedding.weight.copy_(new_weights)

    def calculate_loss(self, embeddings, indices):
        users_emb, items_emb = embeddings
        user_idx, pos_idx, neg_idx = indices
        u_e = users_emb[user_idx]
        p_e = items_emb[pos_idx]
        n_e = items_emb[neg_idx]

        pos_scores = torch.mul(u_e, p_e).sum(dim=1)
        neg_scores = torch.mul(u_e, n_e).sum(dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        reg_loss = 0.5 * (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / float(len(user_idx))

        return loss, reg_loss