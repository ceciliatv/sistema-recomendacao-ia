"""Motor de recomendação: mistura colaborativo, conteúdo e popularidade por domínio-alvo.

Regras (spec 003):
- com histórico no domínio e modelo treinado: a·colaborativo + (1−a)·conteúdo + 0,05·popularidade,
  com a = n/(n+3), n = itens curtidos no domínio;
- sem histórico no domínio (ou domínio só de conteúdo, como séries): conteúdo do perfil inteiro;
- perfil vazio: popularidade (restrita às tags preferidas, se houver).
Os scores viram percentis entre os candidatos antes de somar, para ficarem comparáveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from repertorio.catalog import Catalog
from repertorio.data.schema import DOMAIN_LABELS_PT, Domain, domain_of
from repertorio.models.store import DomainModel
from repertorio.profile import TasteProfile
from repertorio.retrieval.content import ContentIndex, Filters, normalize_tags

POPULARITY_WEIGHT = 0.05
ALPHA_PRIOR = 3.0


def collaborative_weight(n_in_domain: int) -> float:
    return n_in_domain / (n_in_domain + ALPHA_PRIOR)


def _percentile(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return ((rankdata(values, method="average") - 1) / max(len(values) - 1, 1)).astype(np.float32)


@dataclass
class Recommendation:
    item_id: str
    score: float
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)


class RecommendationEngine:
    def __init__(self, catalog: Catalog, content: ContentIndex, models: dict[str, DomainModel] | None = None):
        self.catalog = catalog
        self.content = content
        self.models = models or {}

    # ---------------------------------------------------------------- perfil
    def recommend(
        self, profile: TasteProfile, domain: Domain, k: int = 10, filters: Filters | None = None
    ) -> list[Recommendation]:
        filters = filters or Filters()
        filters.exclude_ids = set(filters.exclude_ids) | profile.consumed
        filters.exclude_tags = list(dict.fromkeys(filters.exclude_tags + normalize_tags(profile.avoided_tags)))

        rows = self.catalog.domain_rows(domain)
        rows = rows[self.content.filter_mask(rows, filters)]
        if profile.preferred_tags and not profile.liked:
            wanted = set(normalize_tags(profile.preferred_tags))
            rows = rows[np.array([bool(self.content.tag_sets[r] & wanted) for r in rows], dtype=bool)]
        if len(rows) == 0:
            return []

        popularity = self.catalog.popularity[rows]
        liked_rows = self.catalog.rows(profile.liked)
        disliked_rows = self.catalog.rows(profile.disliked)
        in_domain = [i for i in profile.liked if domain_of(i) == domain]
        model = self.models.get(domain)
        history = model.to_local(in_domain) if model else np.zeros(0, dtype=np.int64)

        signals: dict[str, np.ndarray] = {}
        vector = None
        if not profile.is_empty():
            vector = self.content.profile_vector(
                liked_rows, disliked_rows, profile.preferred_tags, profile.avoided_tags
            )
            signals["content"] = _percentile(self.content.scores(vector, rows))
        if model is not None and len(history):
            local_scores = model.recommender.score([history])[0]
            row_ids = self.catalog.items["item_id"].to_numpy()[rows]
            raw = np.array([local_scores[model.local[i]] if i in model.local else -np.inf for i in row_ids])
            raw[np.isinf(raw)] = np.nanmin(raw[~np.isinf(raw)]) if (~np.isinf(raw)).any() else 0.0
            signals["collaborative"] = _percentile(raw)

        alpha = collaborative_weight(len(history)) if "collaborative" in signals else 0.0
        if "collaborative" in signals and "content" in signals:
            final = alpha * signals["collaborative"] + (1 - alpha) * signals["content"]
        elif "content" in signals:
            final = signals["content"]
        elif "collaborative" in signals:
            final = signals["collaborative"]
        else:
            final = np.zeros(len(rows), dtype=np.float32)
        final = final + POPULARITY_WEIGHT * popularity

        order = np.argsort(-final, kind="stable")[:k]
        results = []
        for idx in order:
            row = int(rows[idx])
            item_id = self.catalog.item_id(row)
            reasons = self._reasons(item_id, row, vector, model, history, profile, domain, signals)
            results.append(
                Recommendation(
                    item_id=item_id,
                    score=round(float(final[idx]), 4),
                    reasons=reasons,
                    signals={name: round(float(values[idx]), 3) for name, values in signals.items()}
                    | {"popularity": round(float(popularity[idx]), 3), "alpha": round(alpha, 3)},
                )
            )
        return results

    def _reasons(self, item_id, row, vector, model, history, profile, domain, signals) -> list[str]:
        reasons = []
        if "collaborative" in signals and model is not None and item_id in model.local:
            sims = model.recommender.item_similarity(model.local[item_id])
            if sims is not None and len(history):
                anchor = model.item_ids[int(history[np.argmax(sims[history])])]
                reasons.append(f"quem curtiu {self.catalog.get(anchor)['title']} costuma curtir este")
        if vector is not None:
            shared = self.content.shared_tags(vector, row)
            if shared:
                other = {domain_of(i) for i in profile.liked} - {domain}
                origin = (
                    " (a partir do que você curtiu em " + ", ".join(DOMAIN_LABELS_PT[d] for d in sorted(other)) + ")"
                    if other and "collaborative" not in signals
                    else ""
                )
                reasons.append("tags em comum com seu perfil: " + ", ".join(shared) + origin)
        if not reasons:
            reasons.append(f"entre os mais populares de {DOMAIN_LABELS_PT[domain]}s no catálogo")
        return reasons

    # ------------------------------------------------------------- similares
    def similar(
        self, item_id: str, target_domain: Domain | None = None, k: int = 10, filters: Filters | None = None
    ) -> list[Recommendation]:
        source_row = self.catalog.row(item_id)
        source_domain = domain_of(item_id)
        target = target_domain or source_domain
        filters = filters or Filters()
        filters.exclude_ids = set(filters.exclude_ids) | {item_id}

        rows = self.catalog.domain_rows(target)
        rows = rows[self.content.filter_mask(rows, filters)]
        if len(rows) == 0:
            return []
        vector = self.content.item_vector(source_row)
        content = _percentile(self.content.scores(vector, rows))
        final = content
        model = self.models.get(target)
        collaborative = None
        if target == source_domain and model is not None and item_id in model.local:
            sims = model.recommender.item_similarity(model.local[item_id])
            row_ids = self.catalog.items["item_id"].to_numpy()[rows]
            raw = np.array([sims[model.local[i]] if i in model.local else -1.0 for i in row_ids])
            collaborative = _percentile(raw)
            final = 0.5 * collaborative + 0.5 * content
        final = final + POPULARITY_WEIGHT * self.catalog.popularity[rows]

        title = self.catalog.get(item_id)["title"]
        results = []
        for idx in np.argsort(-final, kind="stable")[:k]:
            row = int(rows[idx])
            reasons = []
            if collaborative is not None:
                reasons.append(f"quem curtiu {title} também curtiu este")
            shared = self.content.shared_tags(vector, row)
            if shared:
                reasons.append(f"tags em comum com {title}: " + ", ".join(shared))
            results.append(
                Recommendation(self.catalog.item_id(row), round(float(final[idx]), 4), reasons or ["parecido"])
            )
        return results
