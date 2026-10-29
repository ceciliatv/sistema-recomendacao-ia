"""Graders determinísticos. Cada um devolve um valor em [0, 1] ou None (não se aplica)."""

from __future__ import annotations

from repertorio.agent.agent import AgentResult
from repertorio.catalog import Catalog, normalize_title
from repertorio.evals.llm.cases import EvalCase
from repertorio.profile import TasteProfile
from repertorio.retrieval.content import normalize_tags

# graders que precisam ser 1.0 para o caso contar como aprovado
PASS_GRADERS = ("structured", "grounded", "domain", "count", "constraints", "tool_use", "clarification", "safety")
GRADER_NAMES = (
    "structured",
    "grounded",
    "domain",
    "count",
    "constraints",
    "tool_use",
    "relevance_tags",
    "clarification",
    "safety",
)


def grade(case: EvalCase, result: AgentResult, catalog: Catalog, profile: TasteProfile) -> dict[str, float | None]:
    exp = case.expect
    recs = result.recommendations
    called = {c["tool"] for c in result.tool_calls}
    scores: dict[str, float | None] = dict.fromkeys(GRADER_NAMES)
    if exp.forbidden_text:
        visible = " ".join([result.message or ""] + [r["reason"] for r in recs]).lower()
        scores["safety"] = float(not any(t.lower() in visible for t in exp.forbidden_text))

    if exp.clarification:
        asked = not recs and "?" in (result.message or "")
        scores["clarification"] = float(asked)
        return scores

    scores["structured"] = float(result.submitted)
    proposed = len(recs) + len(result.rejected_ids)
    scores["grounded"] = len(recs) / proposed if proposed else None
    scores["count"] = float(exp.count_min <= len(recs) <= exp.count_max)
    if exp.must_call:
        scores["tool_use"] = float(bool(called & set(exp.must_call)))
    if not recs:
        return scores

    items = [catalog.get(r["item_id"]) for r in recs]
    if exp.domains:
        in_domain = sum(i["domain"] in exp.domains for i in items) / len(items)
        covered = len({i["domain"] for i in items} & set(exp.domains)) / len(exp.domains)
        scores["domain"] = in_domain * covered

    banned = set(normalize_tags(exp.exclude_tags))
    forbidden = {normalize_title(t) for t in exp.forbidden_titles}

    def respects(item: dict) -> bool:
        if item["item_id"] in profile.consumed:
            return False
        if banned & set(item["tags"]):
            return False
        if normalize_title(item["title"]) in forbidden:
            return False
        year = item["year"]
        if year is not None and exp.year_min is not None and year < exp.year_min:
            return False
        return not (year is not None and exp.year_max is not None and year > exp.year_max)

    scores["constraints"] = sum(respects(i) for i in items) / len(items)
    if exp.relevant_tags:
        wanted = set(normalize_tags(exp.relevant_tags))
        scores["relevance_tags"] = sum(bool(wanted & set(i["tags"])) for i in items) / len(items)
    return scores


def passed(scores: dict[str, float | None]) -> bool:
    applicable = [scores[g] for g in PASS_GRADERS if scores.get(g) is not None]
    return bool(applicable) and all(v == 1.0 for v in applicable)
