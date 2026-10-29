"""Roda os casos de eval com um provedor, aplica os graders e escreve o relatório."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime

import numpy as np

from repertorio.agent.agent import RecommenderAgent
from repertorio.agent.providers import get_provider
from repertorio.config import Paths
from repertorio.evals.llm import graders
from repertorio.evals.llm.cases import CATEGORIES, EvalCase, load_cases
from repertorio.evals.llm.judge import CRITERIA, judge, opposite
from repertorio.runtime import load_engine

Log = Callable[[str], None]
DETERMINISTIC = graders.GRADER_NAMES


def select_cases(paths: Paths, only: list[str] | None = None, limit: int | None = None) -> list[EvalCase]:
    cases = load_cases(paths.cases)
    if only:
        cases = [c for c in cases if c.id in only or c.category in only]
    return cases[:limit] if limit else cases


def run_llm_eval(
    paths: Paths,
    provider_name: str,
    model: str | None = None,
    judge_name: str | None = None,
    cases: list[EvalCase] | None = None,
    log: Log = print,
) -> dict:
    """`judge_name`: None desliga o juiz; "auto" usa o provedor oposto."""
    engine = load_engine(paths)
    provider = get_provider(provider_name, model)
    judge_provider = None
    if judge_name:
        judge_provider = get_provider(opposite(provider_name) if judge_name == "auto" else judge_name)
    cases = cases if cases is not None else select_cases(paths)

    records = []
    for case in cases:
        profile = case.build_profile(engine.catalog)
        agent = RecommenderAgent(provider, engine, profile)
        result = agent.ask(case.message)
        scores = graders.grade(case, result, engine.catalog, agent.profile)
        record = {
            "id": case.id,
            "category": case.category,
            "message": case.message,
            "scores": scores,
            "passed": graders.passed(scores),
            "result": asdict(result),
        }
        if judge_provider is not None:
            record["judge"] = judge(judge_provider, case, result, engine.catalog, agent.profile)
        records.append(record)
        mark = "ok " if record["passed"] else "x  "
        titles = ", ".join(r["title"] for r in result.recommendations[:3])
        log(f"{mark}{case.id:<28} {titles[:70]}")

    summary = summarize(records, provider.name, getattr(provider, "model", ""), judge_provider)
    write_outputs(paths, records, summary)
    return summary


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


def summarize(records: list[dict], provider: str, model: str, judge_provider=None) -> dict:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
    graders_mean = {g: _mean([r["scores"][g] for r in records]) for g in DETERMINISTIC}
    judged = [r["judge"] for r in records if r.get("judge")]
    judge_mean = {c: _mean([j[f"judge_{c}"] for j in judged]) for c in CRITERIA} if judged else {}
    usage = [r["result"]["usage"] for r in records]
    return {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "provider": provider,
        "model": model,
        "judge": f"{judge_provider.name}:{judge_provider.model}" if judge_provider else None,
        "cases": len(records),
        "pass_rate": _mean([float(r["passed"]) for r in records]),
        "graders": graders_mean,
        "judge_scores": judge_mean,
        "categories": {
            c: {"cases": len(rs), "pass_rate": _mean([float(r["passed"]) for r in rs])}
            for c in CATEGORIES
            if (rs := by_category.get(c))
        },
        "latency_p50_s": _mean([float(np.median([r["result"]["latency_s"] for r in records]))]),
        "tokens": {
            "input": int(sum(u["input_tokens"] for u in usage)),
            "output": int(sum(u["output_tokens"] for u in usage)),
            "cached_input": int(sum(u["cached_input_tokens"] for u in usage)),
        },
    }


def render_summary(summary: dict) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [
        f"# Eval do agente — {summary['provider']} ({summary['model']})",
        "",
        f"Execução `{summary['run_id']}` · {summary['cases']} casos · juiz: {summary['judge'] or 'desligado'}",
        "",
        f"**Taxa de aprovação:** {fmt(summary['pass_rate'])}",
        "",
        "| Grader | Média |",
        "|---|---:|",
        *[f"| {g} | {fmt(v)} |" for g, v in summary["graders"].items()],
        *[f"| juiz: {c} | {fmt(v)} |" for c, v in summary["judge_scores"].items()],
        "",
        "| Categoria | Casos | Aprovação |",
        "|---|---:|---:|",
        *[f"| {c} | {v['cases']} | {fmt(v['pass_rate'])} |" for c, v in summary["categories"].items()],
        "",
        f"Latência mediana por caso: {summary['latency_p50_s']} s · tokens: {summary['tokens']['input']} de "
        f"entrada ({summary['tokens']['cached_input']} em cache), {summary['tokens']['output']} de saída.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(paths: Paths, records: list[dict], summary: dict) -> None:
    folder = paths.reports / "llm" / f"{summary['run_id']}-{summary['provider']}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "results.jsonl").open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    markdown = render_summary(summary)
    (folder / "summary.md").write_text(markdown)
    # cópia estável da última execução de cada provedor (esta vai para o Git)
    (paths.reports / f"llm_{summary['provider']}.md").write_text(markdown)
    summary["folder"] = str(folder)
