"""Avaliação offline dos modelos de ranking (spec 002)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np

from repertorio.config import Paths
from repertorio.data.schema import INTERACTION_DOMAINS
from repertorio.evals import metrics
from repertorio.models.base import Recommender, Split, leave_one_out, new_users
from repertorio.models.store import load_interactions, make_models

Log = Callable[[str], None]
PROTOCOLS = {"loo": leave_one_out, "new-users": new_users}
METRIC_LABELS = {"hr@10": "HR@10", "recall@20": "Recall@20", "ndcg@10": "NDCG@10", "coverage@10": "Cobertura@10"}


def evaluate(model: Recommender, split: Split, n_items: int, batch: int = 1024) -> dict[str, float]:
    ranked_parts = []
    for start in range(0, len(split.inputs), batch):
        inputs = split.inputs[start : start + batch]
        ranked_parts.append(metrics.top_k(model.score(inputs), 20, inputs))
    ranked = np.concatenate(ranked_parts)
    return {
        "hr@10": metrics.hit_rate(ranked, split.targets, 10),
        "recall@20": metrics.recall(ranked, split.targets, 20),
        "ndcg@10": metrics.ndcg(ranked, split.targets, 10),
        "coverage@10": metrics.coverage(ranked, n_items, 10),
    }


def run_offline(paths: Paths, protocol: str, domains: list[str] | None = None, log: Log = print) -> dict:
    split_fn = PROTOCOLS[protocol]
    results: dict = {"protocol": protocol, "domains": {}}
    for domain in domains or list(INTERACTION_DOMAINS):
        data = load_interactions(paths, domain)
        split = split_fn(data)
        info = {
            "users": len(data.histories),
            "items": data.n_items,
            "positives": data.n_interactions,
            "test_users": len(split.inputs),
            "models": {},
        }
        log(f"[{domain}] {info['users']} usuários · {info['items']} itens · {info['positives']} positivos")
        for name, model in make_models(domain).items():
            started = time.perf_counter()
            model.fit(split.train, data.n_items)
            fit_seconds = time.perf_counter() - started
            scores = evaluate(model, split, data.n_items)
            scores["fit_seconds"] = round(fit_seconds, 1)
            info["models"][name] = scores
            log(f"  {name:<11} " + "  ".join(f"{METRIC_LABELS[k]}={scores[k]:.4f}" for k in METRIC_LABELS))
        results["domains"][domain] = info
    write_report(paths, results)
    return results


def write_report(paths: Paths, results: dict) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    name = f"offline_{results['protocol']}"
    (paths.reports / f"{name}.json").write_text(json.dumps(results, indent=2))
    (paths.reports / f"{name}.md").write_text(render_markdown(results))


def render_markdown(results: dict) -> str:
    lines = [f"# Avaliação offline — protocolo `{results['protocol']}`", ""]
    for domain, info in results["domains"].items():
        lines += [
            f"## {domain}",
            "",
            f"{info['users']} usuários, {info['items']} itens, {info['positives']} positivos; "
            f"{info['test_users']} usuários avaliados.",
            "",
            "| Modelo | " + " | ".join(METRIC_LABELS.values()) + " | Treino (s) |",
            "|---|" + "---:|" * (len(METRIC_LABELS) + 1),
        ]
        best = {k: max(m[k] for m in info["models"].values()) for k in METRIC_LABELS}
        for model, scores in info["models"].items():
            cells = [f"**{scores[k]:.4f}**" if scores[k] == best[k] else f"{scores[k]:.4f}" for k in METRIC_LABELS]
            lines.append(f"| {model} | " + " | ".join(cells) + f" | {scores['fit_seconds']} |")
        lines.append("")
    return "\n".join(lines)
