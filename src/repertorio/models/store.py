"""Configuração por domínio, treino dos modelos de produção e carga dos artefatos."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from repertorio.config import Paths
from repertorio.data.schema import INTERACTION_DOMAINS
from repertorio.models.base import InteractionData, Recommender
from repertorio.models.itemknn import ItemKNNRecommender
from repertorio.models.popularity import PopularityRecommender
from repertorio.models.two_tower import TwoTowerConfig, TwoTowerRecommender

Log = Callable[[str], None]


@dataclass(frozen=True)
class DomainConfig:
    positive_threshold: float | None
    max_users: int | None
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    serving_model: str = "two_tower"  # modelo usado pelo motor; escolhido pela avaliação offline


# serving_model = melhor NDCG@10 no protocolo new-users (reports/offline_new-users.md),
# porque quem conversa com o agente é sempre uma pessoa fora do treino
DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "movie": DomainConfig(4.0, None, TwoTowerConfig(epochs=40), serving_model="itemknn"),
    "book": DomainConfig(4.0, 10_000, TwoTowerConfig(epochs=10), serving_model="two_tower"),
    "music": DomainConfig(None, None, TwoTowerConfig(epochs=30), serving_model="itemknn"),
}


def load_interactions(paths: Paths, domain: str) -> InteractionData:
    config = DOMAIN_CONFIGS[domain]
    frame = pd.read_parquet(paths.processed / f"interactions_{domain}.parquet")
    return InteractionData.from_frame(
        domain, frame, positive_threshold=config.positive_threshold, max_users=config.max_users
    )


def make_models(domain: str) -> dict[str, Recommender]:
    return {
        "popularity": PopularityRecommender(),
        "itemknn": ItemKNNRecommender(),
        "two_tower": TwoTowerRecommender(DOMAIN_CONFIGS[domain].two_tower),
    }


def train_and_save(paths: Paths, domain: str, log: Log = print) -> None:
    """Treina o ItemKNN e o Two-Tower com todos os dados do domínio e salva em `artifacts/`."""
    data = load_interactions(paths, domain)
    folder = paths.artifacts / "models" / domain
    folder.mkdir(parents=True, exist_ok=True)
    log(f"{domain}: {len(data.histories)} usuários, {data.n_items} itens, {data.n_interactions} positivos")
    (folder / "item_ids.json").write_text(json.dumps(data.item_ids))
    knn = ItemKNNRecommender().fit(data.histories, data.n_items)
    sp.save_npz(folder / "itemknn.npz", knn.similarity)
    log(f"{domain}: treinando Two-Tower")
    TwoTowerRecommender(DOMAIN_CONFIGS[domain].two_tower).fit(data.histories, data.n_items, log=log).save(folder)


@dataclass
class DomainModel:
    """Um modelo treinado e o mapeamento entre o índice local dele e as linhas do catálogo."""

    domain: str
    recommender: Recommender
    item_ids: list[str]

    def __post_init__(self) -> None:
        self.local = {item_id: i for i, item_id in enumerate(self.item_ids)}

    def to_local(self, item_ids: list[str]) -> np.ndarray:
        return np.array([self.local[i] for i in item_ids if i in self.local], dtype=np.int64)


def load_domain_model(paths: Paths, domain: str, kind: str | None = None) -> DomainModel | None:
    folder = paths.artifacts / "models" / domain
    if not (folder / "item_ids.json").exists():
        return None
    kind = kind or DOMAIN_CONFIGS[domain].serving_model
    item_ids = json.loads((folder / "item_ids.json").read_text())
    if kind == "two_tower":
        recommender: Recommender = TwoTowerRecommender.load(folder)
    else:
        knn = ItemKNNRecommender()
        knn.similarity = sp.load_npz(folder / "itemknn.npz").tocsr()
        knn.n_items = len(item_ids)
        recommender = knn
    return DomainModel(domain, recommender, item_ids)


def load_all_models(paths: Paths) -> dict[str, DomainModel]:
    models = {}
    for domain in INTERACTION_DOMAINS:
        model = load_domain_model(paths, domain)
        if model is not None:
            models[domain] = model
    return models
