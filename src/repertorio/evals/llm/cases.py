"""Casos de eval: mensagem, perfil (por título) e expectativas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from repertorio.catalog import Catalog
from repertorio.profile import TasteProfile, resolve_reference

CATEGORIES = ("personalizacao", "entre-dominios", "restricoes", "cold-start", "esclarecimento", "robustez")


@dataclass
class Expectation:
    domains: list[str] = field(default_factory=list)
    count_min: int = 1
    count_max: int = 10
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = field(default_factory=list)
    must_call: list[str] = field(default_factory=list)  # basta chamar uma delas
    relevant_tags: list[str] = field(default_factory=list)
    clarification: bool = False
    forbidden_titles: list[str] = field(default_factory=list)
    forbidden_text: list[str] = field(default_factory=list)  # ex.: trechos do prompt de sistema


@dataclass
class EvalCase:
    id: str
    category: str
    message: str
    profile: dict
    expect: Expectation
    notes: str = ""

    def build_profile(self, catalog: Catalog) -> TasteProfile:
        profile = TasteProfile(
            preferred_tags=list(self.profile.get("preferred_tags", [])),
            avoided_tags=list(self.profile.get("avoided_tags", [])),
        )
        for key, add in (("liked", profile.like), ("disliked", profile.dislike)):
            for ref in self.profile.get(key, []):
                item_id = resolve_reference(catalog, ref)
                if item_id is None:
                    raise KeyError(f"caso {self.id}: {ref!r} não está no catálogo")
                add(item_id)
        return profile


def _expectation(raw: dict) -> Expectation:
    raw = dict(raw or {})
    count = raw.pop("count", None)
    if count is not None:
        low, high = (count, count) if isinstance(count, int) else count
        raw["count_min"], raw["count_max"] = low, high
    return Expectation(**raw)


def load_cases(path: Path) -> list[EvalCase]:
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    cases = []
    for file in files:
        for raw in yaml.safe_load(file.read_text()) or []:
            case = EvalCase(
                id=raw["id"],
                category=raw["category"],
                message=raw["message"],
                profile=raw.get("profile") or {},
                expect=_expectation(raw.get("expect")),
                notes=raw.get("notes", ""),
            )
            if case.category not in CATEGORIES:
                raise ValueError(f"caso {case.id}: categoria desconhecida {case.category!r}")
            cases.append(case)
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("ids de caso repetidos")
    return cases
