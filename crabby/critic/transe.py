import math
from typing import Tuple

import torch

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
