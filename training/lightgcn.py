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

        weight_method = cfg.model.get("layer_weighting", "uniform")

        num_weights = self.n_layers + 1
        if weight_method == "uniform":
            weights = torch.ones(num_weights)
        elif weight_method == "gentle":
            layer_decay = cfg.model.get("layer_decay", 0.1)
            weights = torch.ones(num_weights) - torch.arange(num_weights) * layer_decay
        else:
            raise NotImplementedError

        weights = weights / torch.sum(weights)
        self.register_buffer("layer_weights", weights)

        self.reg_weight = cfg.training.reg_weight

        self.gcl_reg = cfg.training.get("gcl_reg", 0.0)
        self.gcl_eps = cfg.training.get("gcl_eps", 0.0)
        self.gcl_temp = cfg.training.get("gcl_temp", 0.0)

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

    def perturbed_forward(self, graph):
        all_emb = self.embedding.weight
        embs = [all_emb]

        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(graph, x)

            noise = torch.rand_like(x)
            noise = F.normalize(noise, dim=1) * self.gcl_eps
            x = x + noise

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

    def calculate_contrastive_loss(self, idx, view1, view2):
        view1 = view1[idx]
        view2 = view2[idx]

        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        pos_score = (view1 * view2).sum(dim=1)
        pos_score = torch.exp(pos_score / self.gcl_temp)

        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.gcl_temp).sum(dim=1)

        loss = torch.mean(-torch.log(pos_score / ttl_score))
        return loss

    def calculate_loss(self, embeddings, indices, graph=None):
        users_emb, items_emb = embeddings
        user_idx, pos_idx, neg_idx = indices
        u_e = users_emb[user_idx]
        p_e = items_emb[pos_idx]
        n_e = items_emb[neg_idx]

        pos_scores = torch.mul(u_e, p_e).sum(dim=1)
        neg_scores = torch.mul(u_e, n_e).sum(dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        reg_loss = 0.5 * (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / float(len(user_idx))
        reg_loss = self.reg_weight * reg_loss

        if self.gcl_reg > 0 and graph is not None:
            cl_loss = torch.tensor(0.0, device=loss.device)

            user_view_1, item_view_1 = self.perturbed_forward(graph)
            user_view_2, item_view_2 = self.perturbed_forward(graph)

            unique_u_idx = torch.unique(user_idx)
            unique_i_idx = torch.unique(pos_idx)

            cl_loss_user = self.calculate_contrastive_loss(unique_u_idx, user_view_1, user_view_2)
            cl_loss_item = self.calculate_contrastive_loss(unique_i_idx, item_view_1, item_view_2)

            cl_loss = self.gcl_reg * (cl_loss_user + cl_loss_item)

            return loss + reg_loss + cl_loss

        else:
            return loss + reg_loss