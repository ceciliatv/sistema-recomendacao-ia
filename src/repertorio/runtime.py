"""Carrega catálogo, índice de conteúdo e modelos uma vez por processo."""

from __future__ import annotations

from functools import lru_cache

from repertorio.catalog import Catalog
from repertorio.config import Paths, get_paths
from repertorio.engine import RecommendationEngine
from repertorio.models.store import load_all_models
from repertorio.retrieval.content import ContentIndex


@lru_cache(maxsize=4)
def load_engine(paths: Paths | None = None) -> RecommendationEngine:
    paths = paths or get_paths()
    catalog = Catalog.load(paths.processed)
    return RecommendationEngine(catalog, ContentIndex(catalog), load_all_models(paths))
