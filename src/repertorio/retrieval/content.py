"""Índice de conteúdo (tags) e busca textual sobre o catálogo.

Os vetores de tags são esparsos de propósito: as tags em comum entre o perfil e o
item são, ao mesmo tempo, o motivo do score e a explicação mostrada para a pessoa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from repertorio.catalog import Catalog
from repertorio.data.schema import Domain
from repertorio.data.tags import normalize_tag


def normalize_tags(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    for tag in tags or []:
        for norm in normalize_tag(tag) or [tag.strip().lower()]:
            if norm and norm not in out:
                out.append(norm)
    return out


@dataclass
class Filters:
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = field(default_factory=list)
    require_tags: list[str] = field(default_factory=list)
    exclude_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.exclude_tags = normalize_tags(self.exclude_tags)
        self.require_tags = normalize_tags(self.require_tags)


class ContentIndex:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        items = catalog.items
        self.tag_lists: list[list[str]] = [list(t) for t in items["tags"]]
        self.tag_sets = [set(t) for t in self.tag_lists]
        self.years = items["year"].astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)

        self.vocab = sorted({t for tags in self.tag_lists for t in tags})
        self.tag_index = {t: i for i, t in enumerate(self.vocab)}
        rows, cols, vals = [], [], []
        for row, (tags, weights) in enumerate(zip(self.tag_lists, items["tag_weights"], strict=True)):
            for tag, weight in zip(tags, weights, strict=True):
                rows.append(row)
                cols.append(self.tag_index[tag])
                vals.append(float(weight))
        raw = sp.csr_matrix((vals, (rows, cols)), shape=(len(items), len(self.vocab)), dtype=np.float32)
        doc_freq = np.bincount(cols, minlength=len(self.vocab))
        self.idf = (np.log((1 + len(items)) / (1 + doc_freq)) + 1.0).astype(np.float32)
        self.tag_matrix = normalize(raw @ sp.diags(self.idf), norm="l2").tocsr()

        texts = [
            " ".join(
                [
                    str(r.title),
                    " ".join(r.creators),
                    " ".join(t.replace("-", " ") for t in r.tags),
                    " ".join(r.tags),
                    str(r.description),
                ]
            )
            for r in items.itertuples(index=False)
        ]
        self.text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode", max_features=300_000
        )
        self.text_matrix = self.text_vectorizer.fit_transform(texts).tocsr()

    # ------------------------------------------------------------ perfis
    def tag_vector(self, tags: list[str], weight: float = 1.0) -> sp.csr_matrix:
        vec = sp.lil_matrix((1, len(self.vocab)), dtype=np.float32)
        for tag in normalize_tags(tags):
            if tag in self.tag_index:
                vec[0, self.tag_index[tag]] = weight * self.idf[self.tag_index[tag]]
        return vec.tocsr()

    def profile_vector(
        self,
        liked_rows: np.ndarray,
        disliked_rows: np.ndarray | None = None,
        preferred_tags: list[str] | None = None,
        avoided_tags: list[str] | None = None,
    ) -> sp.csr_matrix:
        vec = sp.csr_matrix((1, len(self.vocab)), dtype=np.float32)
        if len(liked_rows):
            vec = vec + sp.csr_matrix(self.tag_matrix[liked_rows].sum(axis=0))
        if disliked_rows is not None and len(disliked_rows):
            vec = vec - 0.5 * sp.csr_matrix(self.tag_matrix[disliked_rows].sum(axis=0))
        if preferred_tags:
            vec = vec + self.tag_vector(preferred_tags, weight=0.5 + 0.1 * len(liked_rows))
        if avoided_tags:
            vec = vec - self.tag_vector(avoided_tags)
        return normalize(vec, norm="l2").tocsr()

    def scores(self, vector: sp.csr_matrix, rows: np.ndarray) -> np.ndarray:
        return np.asarray((self.tag_matrix[rows] @ vector.T).todense(), dtype=np.float32).ravel()

    def item_vector(self, row: int) -> sp.csr_matrix:
        return self.tag_matrix[row]

    def shared_tags(self, vector: sp.csr_matrix, row: int, n: int = 3) -> list[str]:
        """Tags que mais contribuem para a similaridade entre o vetor e o item."""
        contrib = self.tag_matrix[row].multiply(vector).tocoo()
        positive = [(v, c) for v, c in zip(contrib.data, contrib.col, strict=True) if v > 0]
        positive.sort(reverse=True)
        return [self.vocab[c] for _, c in positive[:n]]

    # ------------------------------------------------------------ filtros
    def filter_mask(self, rows: np.ndarray, filters: Filters | None) -> np.ndarray:
        mask = np.ones(len(rows), dtype=bool)
        if filters is None:
            return mask
        years = self.years[rows]
        known = ~np.isnan(years)
        domains = self.catalog.domains[rows]
        if filters.year_min is not None:
            # música não tem ano nas fontes; o filtro de ano não se aplica a ela
            mask &= (known & (years >= filters.year_min)) | (~known & (domains == "music"))
        if filters.year_max is not None:
            mask &= (known & (years <= filters.year_max)) | (~known & (domains == "music"))
        if filters.exclude_tags:
            banned = set(filters.exclude_tags)
            mask &= np.array([not (self.tag_sets[r] & banned) for r in rows], dtype=bool)
        if filters.require_tags:
            wanted = set(filters.require_tags)
            mask &= np.array([bool(self.tag_sets[r] & wanted) for r in rows], dtype=bool)
        if filters.exclude_ids:
            ids = self.catalog.items["item_id"].to_numpy()[rows]
            mask &= ~pd.Series(ids).isin(filters.exclude_ids).to_numpy()
        return mask

    # ------------------------------------------------------------ busca
    def search(
        self, query: str, domain: Domain | None = None, filters: Filters | None = None, limit: int = 10
    ) -> list[tuple[int, float]]:
        """Busca textual (título, criadores, tags, descrição) com um leve reforço de popularidade."""
        expanded = [query] + [t.replace("-", " ") for t in normalize_tags(query.split())]
        q = self.text_vectorizer.transform([" ".join(expanded)])
        sims = np.asarray((self.text_matrix @ q.T).todense(), dtype=np.float32).ravel()
        rows = self.catalog.domain_rows(domain) if domain else np.arange(len(sims))
        rows = rows[sims[rows] > 0]
        rows = rows[self.filter_mask(rows, filters)]
        scores = sims[rows] * (0.85 + 0.15 * self.catalog.popularity[rows])
        order = np.argsort(-scores)[:limit]
        return [(int(rows[i]), float(scores[i])) for i in order]
