"""API HTTP do Repertório."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from repertorio import __version__
from repertorio.config import Paths, get_paths
from repertorio.profile import TasteProfile, resolve_reference
from repertorio.retrieval.content import Filters
from repertorio.runtime import load_engine

DomainName = Literal["book", "movie", "series", "music"]


class ProfileIn(BaseModel):
    liked: list[str] = Field(default_factory=list, description="ids ou 'domínio:Título'")
    disliked: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    avoided_tags: list[str] = Field(default_factory=list)


class RecommendationRequest(ProfileIn):
    domain: DomainName
    k: int = Field(default=10, ge=1, le=50)
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = Field(default_factory=list)
    require_tags: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    provider: Literal["anthropic", "openai", "heuristic"] = "anthropic"
    profile: ProfileIn = Field(default_factory=ProfileIn)


def create_app(paths: Paths | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = load_engine(paths or get_paths())
        yield

    app = FastAPI(title="Repertório", version=__version__, lifespan=lifespan)

    def engine(request: Request):
        return request.app.state.engine

    def build_profile(request: Request, data: ProfileIn) -> TasteProfile:
        catalog = engine(request).catalog
        profile = TasteProfile(preferred_tags=data.preferred_tags, avoided_tags=data.avoided_tags)
        missing = []
        for refs, add in ((data.liked, profile.like), (data.disliked, profile.dislike)):
            for ref in refs:
                item_id = resolve_reference(catalog, ref)
                if item_id is None:
                    missing.append(ref)
                else:
                    add(item_id)
        if missing:
            raise HTTPException(404, f"não encontrados no catálogo: {missing}")
        return profile

    @app.get("/health")
    def health(request: Request) -> dict:
        eng = engine(request)
        return {"status": "ok", "items": len(eng.catalog), "models": sorted(eng.models)}

    @app.get("/items/search")
    def search(
        request: Request,
        q: str = Query(min_length=1),
        domain: DomainName | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        limit: int = Query(10, ge=1, le=50),
    ) -> list[dict]:
        eng = engine(request)
        hits = eng.content.search(q, domain, Filters(year_min, year_max), limit)
        return [eng.catalog.summary(eng.catalog.item_id(row)) | {"score": round(s, 4)} for row, s in hits]

    @app.get("/items/{item_id}")
    def item(request: Request, item_id: str) -> dict:
        eng = engine(request)
        if item_id not in eng.catalog:
            raise HTTPException(404, "item não encontrado")
        return eng.catalog.get(item_id)

    @app.get("/items/{item_id}/similar")
    def similar(
        request: Request, item_id: str, target_domain: DomainName | None = None, k: int = Query(10, ge=1, le=50)
    ) -> list[dict]:
        eng = engine(request)
        if item_id not in eng.catalog:
            raise HTTPException(404, "item não encontrado")
        recs = eng.similar(item_id, target_domain, k)
        return [eng.catalog.summary(r.item_id) | {"score": r.score, "reasons": r.reasons} for r in recs]

    @app.post("/recommendations")
    def recommendations(request: Request, body: RecommendationRequest) -> list[dict]:
        eng = engine(request)
        profile = build_profile(request, body)
        filters = Filters(body.year_min, body.year_max, body.exclude_tags, body.require_tags)
        recs = eng.recommend(profile, body.domain, body.k, filters)
        return [
            eng.catalog.summary(r.item_id) | {"score": r.score, "reasons": r.reasons, "signals": r.signals}
            for r in recs
        ]

    @app.post("/chat")
    def chat(request: Request, body: ChatRequest) -> dict:
        from repertorio.agent.agent import RecommenderAgent
        from repertorio.agent.providers import get_provider
        from repertorio.agent.providers.base import ProviderError

        profile = build_profile(request, body.profile)
        try:
            agent = RecommenderAgent(get_provider(body.provider), engine(request), profile)
            result = agent.ask(body.message)
        except ProviderError as exc:
            raise HTTPException(503, str(exc)) from exc
        return asdict(result) | {"profile": agent.profile.to_dict()}

    return app


app = create_app()
