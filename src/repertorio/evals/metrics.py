"""Métricas de ranking com ranking completo (sem amostragem de negativos)."""

from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, k: int, exclude: list[np.ndarray]) -> np.ndarray:
    """Índices dos k maiores scores por linha, em ordem, ignorando os itens de `exclude`."""
    scores = scores.astype(np.float32, copy=True)
    for row, items in enumerate(exclude):
        scores[row, items] = -np.inf
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def hit_rate(ranked: np.ndarray, targets: list[np.ndarray], k: int) -> float:
    hits = [np.isin(ranked[i, :k], t).any() for i, t in enumerate(targets)]
    return float(np.mean(hits)) if hits else 0.0


def recall(ranked: np.ndarray, targets: list[np.ndarray], k: int) -> float:
    values = [np.isin(ranked[i, :k], t).sum() / min(k, len(t)) for i, t in enumerate(targets) if len(t)]
    return float(np.mean(values)) if values else 0.0


def ndcg(ranked: np.ndarray, targets: list[np.ndarray], k: int) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    values = []
    for i, t in enumerate(targets):
        if not len(t):
            continue
        gains = np.isin(ranked[i, :k], t).astype(np.float64)
        ideal = discounts[: min(k, len(t))].sum()
        values.append(float((gains * discounts[: len(gains)]).sum() / ideal))
    return float(np.mean(values)) if values else 0.0


def coverage(ranked: np.ndarray, n_items: int, k: int) -> float:
    """Fração do catálogo que aparece em ao menos um top-k."""
    return float(len(np.unique(ranked[:, :k])) / n_items) if n_items else 0.0
