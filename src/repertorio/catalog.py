"""Catálogo unificado: acesso por id, por domínio e por título."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from repertorio.data.schema import DOMAINS, Domain

_PAREN = re.compile(r"\([^)]*\)")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title.lower()).encode("ascii", "ignore").decode()
    text = _PAREN.sub(" ", text)
    return _NON_WORD.sub(" ", text).strip()


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return list(value)


class Catalog:
    """Envelopa o `items.parquet`. As linhas do DataFrame são a indexação global dos itens."""

    def __init__(self, items: pd.DataFrame):
        self.items = items.reset_index(drop=True)
        self._row = {item_id: i for i, item_id in enumerate(self.items["item_id"])}
        self._domains = self.items["domain"].to_numpy()
        self._popularity = self.items["popularity"].to_numpy(dtype=np.float32)
        self._norm_titles = [normalize_title(t) for t in self.items["title"]]
        self._by_title: dict[tuple[str, str], list[int]] = {}
        for row, (domain, key) in enumerate(zip(self._domains, self._norm_titles, strict=True)):
            self._by_title.setdefault((domain, key), []).append(row)
        self._fuzzy: tuple[TfidfVectorizer, Any] | None = None

    @classmethod
    def load(cls, processed_dir: Path) -> Catalog:
        path = processed_dir / "items.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe; rode `repertorio data build` antes.")
        return cls(pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._row

    @property
    def popularity(self) -> np.ndarray:
        return self._popularity

    @property
    def domains(self) -> np.ndarray:
        return self._domains

    def row(self, item_id: str) -> int:
        return self._row[item_id]

    def rows(self, item_ids: list[str]) -> np.ndarray:
        return np.array([self._row[i] for i in item_ids if i in self._row], dtype=np.int64)

    def item_id(self, row: int) -> str:
        return str(self.items.at[row, "item_id"])

    def domain_rows(self, domain: Domain) -> np.ndarray:
        return np.flatnonzero(self._domains == domain)

    def get(self, item_id: str) -> dict[str, Any]:
        record = self.items.iloc[self._row[item_id]].to_dict()
        for key in ("creators", "tags", "tag_weights"):
            record[key] = _as_list(record.get(key))
        year = record.get("year")
        record["year"] = None if year is None or pd.isna(year) else int(year)
        rating = record.get("rating")
        record["rating"] = None if rating is None or pd.isna(rating) else round(float(rating), 2)
        record["popularity"] = round(float(record["popularity"]), 3)
        return record

    def summary(self, item_id: str, n_tags: int = 6) -> dict[str, Any]:
        """Versão curta do item, usada nos resultados de ferramenta."""
        item = self.get(item_id)
        return {
            "item_id": item["item_id"],
            "domain": item["domain"],
            "title": item["title"],
            "creators": item["creators"][:3],
            "year": item["year"],
            "tags": item["tags"][:n_tags],
        }

    def find_by_title(self, title: str, domain: Domain | None = None, min_similarity: float = 0.55) -> str | None:
        """Acha um item pelo título: primeiro exato (normalizado), depois aproximado.

        Com vários itens de mesmo título, fica o mais popular.
        """
        key = normalize_title(title)
        domains = [domain] if domain else list(DOMAINS)
        exact = [row for d in domains for row in self._by_title.get((d, key), [])]
        if exact:
            best = max(exact, key=lambda r: self._popularity[r])
            return self.item_id(best)
        vectorizer, matrix = self._fuzzy_index()
        sims = (matrix @ vectorizer.transform([key]).T).toarray().ravel()
        if domain:
            sims[self._domains != domain] = 0.0
        # desempate leve por popularidade
        best = int(np.argmax(sims + 0.01 * self._popularity))
        return self.item_id(best) if sims[best] >= min_similarity else None

    def _fuzzy_index(self) -> tuple[TfidfVectorizer, Any]:
        if self._fuzzy is None:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
            self._fuzzy = (vectorizer, vectorizer.fit_transform(self._norm_titles))
        return self._fuzzy
