"""Perfil de gosto: o que a pessoa curtiu, rejeitou e quais tags prefere ou evita."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from repertorio.catalog import Catalog
from repertorio.data.schema import DOMAINS


def resolve_reference(catalog: Catalog, ref: str) -> str | None:
    """Aceita um item_id ("book:123"), "domínio:Título" ("book:Dune") ou só um título."""
    ref = ref.strip()
    if ref in catalog:
        return ref
    prefix, _, rest = ref.partition(":")
    if rest and prefix in DOMAINS:
        return catalog.find_by_title(rest, prefix)  # type: ignore[arg-type]
    return catalog.find_by_title(ref)


@dataclass
class TasteProfile:
    liked: list[str] = field(default_factory=list)
    disliked: list[str] = field(default_factory=list)
    preferred_tags: list[str] = field(default_factory=list)
    avoided_tags: list[str] = field(default_factory=list)

    @property
    def consumed(self) -> set[str]:
        return set(self.liked) | set(self.disliked)

    def is_empty(self) -> bool:
        return not (self.liked or self.disliked or self.preferred_tags)

    def like(self, item_id: str) -> None:
        if item_id in self.disliked:
            self.disliked.remove(item_id)
        if item_id not in self.liked:
            self.liked.append(item_id)

    def dislike(self, item_id: str) -> None:
        if item_id in self.liked:
            self.liked.remove(item_id)
        if item_id not in self.disliked:
            self.disliked.append(item_id)

    def prefer_tags(self, tags: list[str]) -> None:
        for tag in tags:
            if tag in self.avoided_tags:
                self.avoided_tags.remove(tag)
            if tag not in self.preferred_tags:
                self.preferred_tags.append(tag)

    def avoid_tags(self, tags: list[str]) -> None:
        for tag in tags:
            if tag in self.preferred_tags:
                self.preferred_tags.remove(tag)
            if tag not in self.avoided_tags:
                self.avoided_tags.append(tag)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TasteProfile:
        return cls(**{k: list(data.get(k) or []) for k in ("liked", "disliked", "preferred_tags", "avoided_tags")})

    @classmethod
    def from_references(cls, catalog: Catalog, liked: list[str], disliked: list[str] | None = None) -> TasteProfile:
        """Monta um perfil a partir de ids ou títulos. Referência não encontrada gera erro."""
        profile = cls()
        for ref, add in [(r, profile.like) for r in liked] + [(r, profile.dislike) for r in disliked or []]:
            item_id = resolve_reference(catalog, ref)
            if item_id is None:
                raise KeyError(f"item não encontrado no catálogo: {ref!r}")
            add(item_id)
        return profile

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> TasteProfile:
        return cls.from_dict(json.loads(path.read_text())) if path.exists() else cls()
