"""ItemKNN: similaridade de cosseno item-item sobre a matriz binária de interações."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from repertorio.models.base import Recommender


def histories_to_csr(histories: list[np.ndarray], n_items: int) -> sp.csr_matrix:
    indptr = np.zeros(len(histories) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(h) for h in histories])
    indices = np.concatenate(histories) if histories else np.zeros(0, dtype=np.int64)
    data = np.ones(len(indices), dtype=np.float32)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(len(histories), n_items))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix


class ItemKNNRecommender(Recommender):
    name = "itemknn"

    def __init__(self, k: int = 200, shrink: float = 10.0, block: int = 2048):
        super().__init__()
        self.k = k
        self.shrink = shrink
        self.block = block

    def fit(self, histories: list[np.ndarray], n_items: int) -> ItemKNNRecommender:
        self.n_items = n_items
        x = histories_to_csr(histories, n_items).tocsc()
        xt = x.T.tocsr()
        norms = np.sqrt(np.asarray(x.sum(axis=0)).ravel()).astype(np.float32)
        rows, cols, vals = [], [], []
        # calcula a similaridade em blocos de itens para não materializar n_items × n_items
        for start in range(0, n_items, self.block):
            stop = min(start + self.block, n_items)
            co = (xt[start:stop] @ x).toarray().astype(np.float32)
            denom = np.outer(norms[start:stop], norms) + self.shrink + 1e-8
            sim = co / denom
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0  # sem auto-similaridade
            k = min(self.k, n_items - 1)
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            top_vals = np.take_along_axis(sim, top, axis=1)
            keep = top_vals > 0
            rows.append(np.repeat(np.arange(start, stop), k)[keep.ravel()])
            cols.append(top[keep])
            vals.append(top_vals[keep])
        self.similarity = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items)
        )
        return self

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        user_matrix = histories_to_csr(histories, self.n_items)
        return np.asarray((user_matrix @ self.similarity).todense(), dtype=np.float32)

    def item_similarity(self, item: int) -> np.ndarray:
        return np.asarray(self.similarity[item].todense(), dtype=np.float32).ravel()
