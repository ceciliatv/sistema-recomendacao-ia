"""Dados de interação, splits de avaliação e a interface comum dos modelos.

Todos os modelos recebem **históricos** (arrays de índices de item) em vez de ids de
usuário. Assim eles são indutivos: funcionam para quem não estava no treino.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class InteractionData:
    domain: str
    item_ids: list[str]  # índice local -> item_id
    user_ids: list[str]
    histories: list[np.ndarray]  # por usuário, índices locais em ordem (temporal quando houver)
    temporal: bool

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_interactions(self) -> int:
        return int(sum(len(h) for h in self.histories))

    @classmethod
    def from_frame(
        cls,
        domain: str,
        frame: pd.DataFrame,
        positive_threshold: float | None,
        min_user: int = 5,
        min_item: int = 3,
        max_users: int | None = None,
        seed: int = 42,
    ) -> InteractionData:
        """Filtra positivos, amostra usuários (opcional) e aplica k-core até estabilizar."""
        df = frame
        if positive_threshold is not None:
            df = df[df["value"] >= positive_threshold]
        df = df.drop_duplicates(["user_id", "item_id"])
        if max_users is not None:
            users = np.sort(df["user_id"].unique())
            if len(users) > max_users:
                keep = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
                df = df[df["user_id"].isin(keep)]
        while True:
            size = len(df)
            item_counts = df["item_id"].value_counts()
            df = df[df["item_id"].isin(item_counts[item_counts >= min_item].index)]
            user_counts = df["user_id"].value_counts()
            df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
            if len(df) == size:
                break
        temporal = bool((df["timestamp"] > 0).any())
        df = df.sort_values(["user_id", "timestamp"] if temporal else ["user_id"], kind="stable")
        item_ids = sorted(df["item_id"].unique())
        local = {item_id: i for i, item_id in enumerate(item_ids)}
        df = df.assign(local=df["item_id"].map(local).astype(np.int64))
        histories, user_ids = [], []
        for user_id, group in df.groupby("user_id", sort=True):
            user_ids.append(str(user_id))
            histories.append(group["local"].to_numpy())
        return cls(domain, item_ids, user_ids, histories, temporal)


@dataclass
class Split:
    train: list[np.ndarray]  # históricos usados no fit
    inputs: list[np.ndarray]  # histórico que o modelo vê na hora de recomendar
    targets: list[np.ndarray]  # itens escondidos que deveriam aparecer


def leave_one_out(data: InteractionData, seed: int = 42) -> Split:
    """Esconde um positivo por usuário: o último (temporal) ou um sorteado."""
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for history in data.histories:
        pick = len(history) - 1 if data.temporal else int(rng.integers(len(history)))
        targets.append(history[pick : pick + 1])
        inputs.append(np.delete(history, pick))
    return Split(train=inputs, inputs=inputs, targets=targets)


def new_users(data: InteractionData, test_fraction: float = 0.2, seed: int = 42) -> Split:
    """Separa usuários inteiros: os de teste nunca aparecem no treino.

    Metade do histórico de cada usuário de teste vira entrada e a outra metade é o alvo.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.histories))
    n_test = int(len(order) * test_fraction)
    test_users, train_users = order[:n_test], order[n_test:]
    train = [data.histories[u] for u in train_users]
    inputs, targets = [], []
    for u in test_users:
        history = rng.permutation(data.histories[u])
        cut = max(1, len(history) // 2)
        inputs.append(history[:cut])
        targets.append(history[cut:])
    return Split(train=train, inputs=inputs, targets=targets)


class Recommender(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.n_items = 0

    @abstractmethod
    def fit(self, histories: list[np.ndarray], n_items: int) -> Recommender: ...

    @abstractmethod
    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """Scores (len(histories) × n_items); maior é melhor. Itens do histórico não são mascarados aqui."""

    def item_similarity(self, item: int) -> np.ndarray | None:
        """Similaridade do item com todos os outros, quando o modelo sabe calcular."""
        return None
