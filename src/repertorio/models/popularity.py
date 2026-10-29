"""Linha de base: recomenda os itens com mais interações positivas."""

from __future__ import annotations

import numpy as np

from repertorio.models.base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def fit(self, histories: list[np.ndarray], n_items: int) -> PopularityRecommender:
        self.n_items = n_items
        counts = np.bincount(np.concatenate(histories), minlength=n_items).astype(np.float32)
        self.counts = counts / max(counts.max(), 1.0)
        return self

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        return np.tile(self.counts, (len(histories), 1))
