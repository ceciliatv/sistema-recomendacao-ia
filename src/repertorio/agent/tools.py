"""Ferramentas do agente: esquemas, validação e execução sobre o motor.

O `ToolBox` registra todo item_id que apareceu num resultado de ferramenta. Uma
recomendação só é aceita se o id estiver nesse conjunto (ancoragem): ids que o
modelo inventou ou lembrou de fora do catálogo são descartados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from repertorio.agent.providers.base import ToolCall, ToolResult, ToolSpec
from repertorio.data.schema import DOMAINS
from repertorio.engine import Recommendation, RecommendationEngine
from repertorio.profile import TasteProfile, resolve_reference
from repertorio.retrieval.content import Filters

DomainName = Literal["book", "movie", "series", "music"]

# ------------------------------------------------------------------ argumentos


class SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    domain: DomainName | None = None
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=20)


class ItemArgs(BaseModel):
    item_id: str


class EmptyArgs(BaseModel):
    pass


class RecommendArgs(BaseModel):
    domain: DomainName
    k: int = Field(default=8, ge=1, le=20)
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = Field(default_factory=list)
    require_tags: list[str] = Field(default_factory=list)


class SimilarArgs(BaseModel):
    item_id: str
    target_domain: DomainName | None = None
    k: int = Field(default=8, ge=1, le=20)
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = Field(default_factory=list)


class UpdateProfileArgs(BaseModel):
    liked: list[str] = Field(default_factory=list)
    disliked: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    avoided_tags: list[str] = Field(default_factory=list)


class Pick(BaseModel):
    item_id: str
    reason: str = ""


class SubmitArgs(BaseModel):
    recommendations: list[Pick] = Field(max_length=20)
    message: str = ""


# ------------------------------------------------------------------ esquemas

_DOMAIN = {"type": "string", "enum": list(DOMAINS), "description": "book, movie, series ou music"}
_YEAR_MIN = {"type": "integer", "description": "Ano mínimo de lançamento (inclusive)."}
_YEAR_MAX = {"type": "integer", "description": "Ano máximo de lançamento (inclusive)."}
_EXCLUDE = {"type": "array", "items": {"type": "string"}, "description": 'Tags proibidas, ex.: ["horror"].'}
_K = {"type": "integer", "minimum": 1, "maximum": 20, "description": "Quantos itens devolver (padrão 8)."}


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "search_catalog",
        "Busca no catálogo por texto livre: título, autor/artista, tema, humor ou gênero. "
        "Use para achar um item citado pela pessoa ou itens sobre um tema. Devolve itens com item_id.",
        _obj(
            {
                "query": {"type": "string", "description": "Texto da busca, de preferência em inglês."},
                "domain": _DOMAIN,
                "year_min": _YEAR_MIN,
                "year_max": _YEAR_MAX,
                "exclude_tags": _EXCLUDE,
                "limit": _K,
            },
            ["query"],
        ),
    ),
    ToolSpec(
        "get_item_details",
        "Detalhes de um item do catálogo (tags, ano, criadores, descrição, nota, popularidade).",
        _obj({"item_id": {"type": "string"}}, ["item_id"]),
    ),
    ToolSpec(
        "get_user_profile",
        "Perfil de gosto da pessoa: itens curtidos e rejeitados, tags preferidas e evitadas e as tags "
        "mais fortes do perfil.",
        _obj({}),
    ),
    ToolSpec(
        "recommend_for_user",
        "Recomendações personalizadas num domínio a partir do perfil. Combina filtragem colaborativa, "
        "tags de conteúdo e popularidade, e funciona mesmo sem histórico no domínio (usa o perfil dos "
        "outros domínios). Já exclui itens do perfil. Cada item vem com os motivos.",
        _obj(
            {
                "domain": _DOMAIN,
                "k": _K,
                "year_min": _YEAR_MIN,
                "year_max": _YEAR_MAX,
                "exclude_tags": _EXCLUDE,
                "require_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Só itens com pelo menos uma destas tags.",
                },
            },
            ["domain"],
        ),
    ),
    ToolSpec(
        "find_similar",
        "Itens parecidos com um item específico, no mesmo domínio ou em outro (target_domain). "
        "Use quando o pedido é 'parecido com X'.",
        _obj(
            {
                "item_id": {"type": "string"},
                "target_domain": _DOMAIN,
                "k": _K,
                "year_min": _YEAR_MIN,
                "year_max": _YEAR_MAX,
                "exclude_tags": _EXCLUDE,
            },
            ["item_id"],
        ),
    ),
    ToolSpec(
        "update_profile",
        "Atualiza o perfil com o que a pessoa disse que curtiu ou não curtiu (item_ids ou títulos no "
        "formato 'domínio:Título', ex.: 'book:Dune') e com tags que ela prefere ou quer evitar.",
        _obj(
            {
                "liked": {"type": "array", "items": {"type": "string"}},
                "disliked": {"type": "array", "items": {"type": "string"}},
                "preferred_tags": {"type": "array", "items": {"type": "string"}},
                "avoided_tags": {"type": "array", "items": {"type": "string"}},
            }
        ),
    ),
    ToolSpec(
        "submit_recommendations",
        "Entrega a resposta final. Use só item_ids que vieram de outras ferramentas nesta conversa. "
        "Cada item leva um motivo de uma frase, específico para esta pessoa. `message` é o texto curto "
        "para a pessoa, no idioma dela.",
        _obj(
            {
                "recommendations": {
                    "type": "array",
                    "maxItems": 20,
                    "items": _obj({"item_id": {"type": "string"}, "reason": {"type": "string"}}, ["item_id", "reason"]),
                },
                "message": {"type": "string"},
            },
            ["recommendations", "message"],
        ),
    ),
]

_ARGS = {
    "search_catalog": SearchArgs,
    "get_item_details": ItemArgs,
    "get_user_profile": EmptyArgs,
    "recommend_for_user": RecommendArgs,
    "find_similar": SimilarArgs,
    "update_profile": UpdateProfileArgs,
    "submit_recommendations": SubmitArgs,
}

# ------------------------------------------------------------------ execução


@dataclass
class Submission:
    picks: list[dict[str, Any]]  # aceitos: item_id, reason, title, domain
    rejected: list[str]  # ids não ancorados
    message: str


@dataclass
class ToolBox:
    engine: RecommendationEngine
    profile: TasteProfile
    seen_ids: set[str] = field(default_factory=set)
    trace: list[dict[str, Any]] = field(default_factory=list)
    submission: Submission | None = None

    @property
    def specs(self) -> list[ToolSpec]:
        return TOOL_SPECS

    @property
    def catalog(self):
        return self.engine.catalog

    def new_turn(self) -> None:
        self.submission = None
        self.trace = []

    def execute(self, call: ToolCall) -> ToolResult:
        record: dict[str, Any] = {"tool": call.name, "arguments": call.arguments}
        self.trace.append(record)
        if call.invalid_json is not None:
            record["error"] = "json inválido"
            return ToolResult(call.id, "argumentos não são JSON válido", is_error=True)
        model = _ARGS.get(call.name)
        if model is None:
            record["error"] = "ferramenta desconhecida"
            return ToolResult(call.id, f"ferramenta desconhecida: {call.name}", is_error=True)
        try:
            args = model.model_validate(call.arguments)
        except ValidationError as exc:
            record["error"] = "validação"
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolResult(call.id, f"argumentos inválidos: {problems}", is_error=True)
        try:
            payload = getattr(self, f"_{call.name}")(args)
        except KeyError as exc:
            record["error"] = str(exc)
            return ToolResult(call.id, f"não encontrado: {exc}", is_error=True)
        ids = [i["item_id"] for i in payload.get("items", [])]
        self.seen_ids.update(ids)
        record["result_ids"] = ids
        return ToolResult(call.id, json.dumps(payload, ensure_ascii=False))

    # ---------------------------------------------------------- ferramentas
    def _item(self, item_id: str, extra: dict | None = None) -> dict:
        return self.catalog.summary(item_id) | (extra or {})

    def _recs(self, recs: list[Recommendation]) -> dict:
        return {"items": [self._item(r.item_id, {"reasons": r.reasons, "score": r.score}) for r in recs]}

    def _search_catalog(self, args: SearchArgs) -> dict:
        filters = Filters(args.year_min, args.year_max, args.exclude_tags)
        hits = self.engine.content.search(args.query, args.domain, filters, args.limit)
        return {"items": [self._item(self.catalog.item_id(row), {"score": round(s, 3)}) for row, s in hits]}

    def _get_item_details(self, args: ItemArgs) -> dict:
        if args.item_id not in self.catalog:
            raise KeyError(f"{args.item_id} não está no catálogo; use search_catalog")
        item = self.catalog.get(args.item_id)
        details = {
            "description": item["description"][:400],
            "rating": item["rating"],
            "popularity": item["popularity"],
            "tags": item["tags"][:12],
        }
        return {"items": [self._item(args.item_id, details)]}

    def _get_user_profile(self, _: EmptyArgs) -> dict:
        profile = self.profile
        liked_rows = self.catalog.rows(profile.liked)
        top_tags: list[str] = []
        if len(liked_rows):
            vector = self.engine.content.profile_vector(liked_rows)
            weights = vector.toarray().ravel()
            top_tags = [self.engine.content.vocab[i] for i in weights.argsort()[::-1][:8] if weights[i] > 0]
        return {
            "liked": [self.catalog.summary(i, n_tags=4) for i in profile.liked],
            "disliked": [self.catalog.summary(i, n_tags=4) for i in profile.disliked],
            "preferred_tags": profile.preferred_tags,
            "avoided_tags": profile.avoided_tags,
            "top_profile_tags": top_tags,
            "is_empty": profile.is_empty(),
        }

    def _recommend_for_user(self, args: RecommendArgs) -> dict:
        filters = Filters(args.year_min, args.year_max, args.exclude_tags, args.require_tags)
        return self._recs(self.engine.recommend(self.profile, args.domain, args.k, filters))

    def _find_similar(self, args: SimilarArgs) -> dict:
        if args.item_id not in self.catalog:
            raise KeyError(f"{args.item_id} não está no catálogo; use search_catalog")
        filters = Filters(args.year_min, args.year_max, args.exclude_tags)
        filters.exclude_ids = self.profile.consumed
        return self._recs(self.engine.similar(args.item_id, args.target_domain, args.k, filters))

    def _update_profile(self, args: UpdateProfileArgs) -> dict:
        resolved, missing = [], []
        for refs, add in ((args.liked, self.profile.like), (args.disliked, self.profile.dislike)):
            for ref in refs:
                item_id = resolve_reference(self.catalog, ref)
                if item_id is None:
                    missing.append(ref)
                else:
                    add(item_id)
                    resolved.append(item_id)
        self.profile.prefer_tags(args.preferred_tags)
        self.profile.avoid_tags(args.avoided_tags)
        return {
            "items": [self._item(i) for i in resolved],
            "not_found": missing,
            "preferred_tags": self.profile.preferred_tags,
            "avoided_tags": self.profile.avoided_tags,
        }

    def _submit_recommendations(self, args: SubmitArgs) -> dict:
        picks, rejected, used = [], [], set()
        for pick in args.recommendations:
            if pick.item_id in used:
                continue
            used.add(pick.item_id)
            if pick.item_id in self.catalog and pick.item_id in self.seen_ids:
                item = self.catalog.get(pick.item_id)
                picks.append(
                    {"item_id": pick.item_id, "reason": pick.reason, "title": item["title"], "domain": item["domain"]}
                )
            else:
                rejected.append(pick.item_id)
        self.submission = Submission(picks, rejected, args.message)
        # não devolve "items": a entrega não acrescenta ids ao conjunto ancorado
        return {"accepted": len(picks), "rejected_not_grounded": rejected}
