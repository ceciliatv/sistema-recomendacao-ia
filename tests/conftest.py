"""Catálogo sintético pequeno: os testes não acessam a rede nem os datasets reais."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repertorio.catalog import Catalog
from repertorio.config import Paths
from repertorio.engine import RecommendationEngine
from repertorio.models import store
from repertorio.models.two_tower import TwoTowerConfig
from repertorio.retrieval.content import ContentIndex

# três "gostos" que atravessam os domínios pelas tags em comum
CLUSTERS = {
    "scifi": ["science-fiction", "dystopia", "dark"],
    "romance": ["romance", "comedy", "feel-good"],
    "horror": ["horror", "thriller", "dark"],
}
MUSIC_TAGS = {
    "scifi": ["electronic", "dark", "synthwave"],
    "romance": ["pop", "romance", "feel-good"],
    "horror": ["metal", "horror", "dark"],
}
DOMAINS = ("book", "movie", "series", "music")
PER_CLUSTER = 6


def make_items() -> pd.DataFrame:
    records = []
    for domain in DOMAINS:
        for cluster, tags in CLUSTERS.items():
            for i in range(PER_CLUSTER):
                item_tags = MUSIC_TAGS[cluster] if domain == "music" else tags
                records.append(
                    {
                        "item_id": f"{domain}:{cluster}{i}",
                        "domain": domain,
                        "source_id": f"{cluster}{i}",
                        "title": f"{cluster.title()} {domain.title()} {i}",
                        "creators": [f"Autora {cluster} {i % 2}"],
                        "year": None if domain == "music" else 1990 + 5 * i,
                        "tags": item_tags[: 2 + i % 2],
                        "tag_weights": [1.0, 0.8, 0.5][: 2 + i % 2],
                        "description": f"Um {domain} de {cluster}.",
                        "popularity": (i + 1) / PER_CLUSTER,
                        "rating": 4.0,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame["year"] = frame["year"].astype("Int64")
    frame["popularity"] = frame["popularity"].astype("float32")
    return frame


def make_interactions(domain: str, n_users: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = list(CLUSTERS)
    rows = []
    for user in range(n_users):
        cluster = clusters[user % len(clusters)]
        picks = rng.choice(PER_CLUSTER, size=int(rng.integers(4, PER_CLUSTER + 1)), replace=False)
        for t, i in enumerate(picks):
            rows.append((f"u{user}", f"{domain}:{cluster}{i}", 5.0, t + 1))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "value", "timestamp"])


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Paths:
    paths = Paths(tmp_path_factory.mktemp("repertorio"))
    paths.processed.mkdir(parents=True)
    make_items().to_parquet(paths.processed / "items.parquet", index=False)
    for domain in ("book", "movie", "music"):
        make_interactions(domain).to_parquet(paths.processed / f"interactions_{domain}.parquet", index=False)
    fast = TwoTowerConfig(dim=16, epochs=25, batch_size=64, lr=1e-2)
    for domain in ("book", "movie", "music"):
        store.DOMAIN_CONFIGS[domain] = store.DomainConfig(positive_threshold=4.0, max_users=None, two_tower=fast)
        store.train_and_save(paths, domain, log=lambda _: None)
    return paths


@pytest.fixture(scope="session")
def catalog(project) -> Catalog:
    return Catalog.load(project.processed)


@pytest.fixture(scope="session")
def content(catalog) -> ContentIndex:
    return ContentIndex(catalog)


@pytest.fixture(scope="session")
def engine(project, catalog, content) -> RecommendationEngine:
    return RecommendationEngine(catalog, content, store.load_all_models(project))
