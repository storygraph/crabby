import math
from typing import Tuple

import torch
import torch.utils.data as torch_data

import crabby.critic.data as data


class TranseModel(torch.nn.Module):
    def __init__(self, onto: data.Ontology, k: int):
        super(TranseModel, self).__init__()

        self.entity_embeddings = self._entity_embeddings(onto.entities_len(), k)
        self.rel_embeddings = self._rel_embeddings(onto.relations_len(), k)

    def forward(self, x):
        head_emb = self.entity_embeddings(x[:, 0])
        rel_emb = self.rel_embeddings(x[:, 1])
        tail_emb = self.entity_embeddings(x[:, 2])

        return (head_emb + rel_emb - tail_emb).pow(2).sum(1).sqrt()

    def _entity_embeddings(self, num_entities: int, k: int) -> torch.nn.Embedding:
        high, low = self._initial_boundaries(k)
        entity_tensor = (high - low) * torch.rand(num_entities, k) + low

        # Normalize entity embeddings to prevent a trivial optimisation of the loss function.
        return torch.nn.Embedding.from_pretrained(entity_tensor, freeze=False, max_norm=1)

    def _rel_embeddings(self, num_rels: int, k: int) -> torch.nn.Embedding:
        high, low = self._initial_boundaries(k)
        rel_tensor = (high - low) * torch.rand(num_rels, k) + low
        
        for _, rel_emb in enumerate(rel_tensor):
            rel_emb /= torch.linalg.norm(rel_emb, ord=2)
        
        return torch.nn.Embedding.from_pretrained(rel_tensor, freeze=False)

    def _initial_boundaries(self, k: int) -> Tuple[float, float]:
        return -6.0 / math.sqrt(k), 6.0 / math.sqrt(k)


class Trainer:
    _training_loader: torch_data.DataLoader
    _onto: data.Ontology
    _optimizer: torch.optim.Optimizer
    _model: TranseModel
    _margin: float
    _epoch: int

    def __init__(
        self, 
        training_loader: torch_data.DataLoader,
        onto: data.Ontology,
        optimizer: torch.optim.Optimizer,
        model: TranseModel,
        margin: float,
    ) -> None:
        self._training_loader = training_loader
        self._onto = onto
        self._optimizer = optimizer
        self._model = model
        self._margin = margin

        self._epoch = 0

    def train_one_epoch(self) -> None:
        self._epoch += 1
        minibatch_count = 0
        cum_loss = 0

        # Sample a minibatch of triplets on each turn.
        for _, triplets in enumerate(self._training_loader):
            self._optimizer.zero_grad()
            
            corrupted_triplets = data.corrupted_counterparts(self._onto, triplets)
            
            out = self._model(triplets)
            corrupted_out = self._model(corrupted_triplets)
            
            loss = torch.nn.functional.relu(self._margin + out - corrupted_out).sum()
            loss.backward()
            
            # Adjust learning weights
            self._optimizer.step()

            cum_loss += loss / len(triplets)
            minibatch_count += 1
        
        print(f"[Epoch {self._epoch}] Average loss ---> {cum_loss / minibatch_count}")

    def epoch(self) -> int:
        return self._epoch

    def model(self) -> TranseModel:
        return self._model
