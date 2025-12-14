import json

import torch
import tqdm

from training.dataset import *
from training.evaluation import *

class Trainer():
    def __init__(self, cfg, model):
        self.cfg = cfg

        self.device = torch.device(cfg.training.device)
        self.n_epochs = cfg.training.n_epochs

        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

        self.graph = load_adjacency_matrix(cfg).to(self.device)
        self.datasets = construct_datasets(cfg)

        self.evaluator = Evaluator(self.cfg.evaluation, "valid", self.datasets)

    def train(self):
        os.makedirs(self.cfg.paths.embedding, exist_ok=True)

        train_dataloder = self.datasets["train"]
        train_losses = []
        valid_metrics = {"recall": [], "ndcg": []}

        best_ndcg = 0
        n_patience = 0
        for n_epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm.tqdm(train_dataloder, desc="Training Model")
            for batch_data in progress_bar:
                batch_data  = [data.to(self.device) for data in batch_data]

                self.optim.zero_grad()

                embeddings = self.model(self.graph)
                bpr_loss, reg_loss = self.model.calculate_loss(embeddings, batch_data)

                loss = bpr_loss + self.cfg.training.reg_weight * reg_loss

                loss.backward()
                self.optim.step()

                running_loss += loss.item() * batch_data[0].size(0)

                progress_bar.set_description(f"Training Model ({loss.item():.6f})")

            train_losses.append(running_loss / len(train_dataloder.dataset))

            valid_recall, valid_ndcg = self.evaluate()
            valid_metrics["recall"].append(valid_recall)
            valid_metrics["ndcg"].append(valid_ndcg)

            print(f"Epoch [{n_epoch + 1:3d}] train loss: {train_losses[-1]:.6f} / valid recall {valid_recall:.6f} / valid ndcg {valid_ndcg:.6f}")

            if valid_ndcg > best_ndcg:
                best_ndcg = valid_ndcg
                n_patience = 0
                torch.save(embeddings, f"{self.cfg.paths.embedding}/{self.cfg.name}.pt")
            else:
                n_patience += 1
                if n_patience == self.cfg.training.early_stopping:
                    print(f"Early stopping at epoch {n_epoch + 1:3d}")
                    break

        os.makedirs(f"{self.cfg.paths.logs}", exist_ok=True)
        with open(f"{self.cfg.paths.logs}/{self.cfg.name}.json", "w") as f:
            json.dump({"train": train_losses, "valid": valid_metrics}, f)

    def get_embeddings(self):
        self.model.eval()

        with torch.no_grad():
            embeddings = self.model(self.graph)

        return embeddings

    def evaluate(self):
        embeddings = self.get_embeddings()
        recall, ndcgs = self.evaluator.evaluate(embeddings)
        return recall, ndcgs