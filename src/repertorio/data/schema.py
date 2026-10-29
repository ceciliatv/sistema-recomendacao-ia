"""Esquema único do catálogo e das interações."""

from __future__ import annotations

from typing import Literal

Domain = Literal["book", "movie", "series", "music"]
DOMAINS: tuple[Domain, ...] = ("book", "movie", "series", "music")

# domínios com interações públicas (os outros são só de conteúdo)
INTERACTION_DOMAINS: tuple[Domain, ...] = ("book", "movie", "music")

DOMAIN_LABELS_PT = {"book": "livro", "movie": "filme", "series": "série", "music": "música"}

ITEM_COLUMNS = [
    "item_id",  # "<domínio>:<id na fonte>"
    "domain",
    "source_id",
    "title",
    "creators",  # lista de autores, diretores ou redes
    "year",  # Int64, pode faltar
    "tags",  # lista de tags normalizadas, da mais forte para a mais fraca
    "tag_weights",  # pesos das tags, mesma ordem, em (0, 1]
    "description",
    "popularity",  # percentil dentro do domínio, 0-1
    "rating",  # nota média da fonte, normalizada para 0-5 (pode faltar)
]

INTERACTION_COLUMNS = ["user_id", "item_id", "value", "timestamp"]


def make_item_id(domain: Domain, source_id: object) -> str:
    return f"{domain}:{source_id}"


def domain_of(item_id: str) -> Domain:
    domain = item_id.split(":", 1)[0]
    if domain not in DOMAINS:
        raise ValueError(f"item_id com domínio desconhecido: {item_id!r}")
    return domain  # type: ignore[return-value]
