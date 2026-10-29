import json

from repertorio.evals.llm.runner import run_llm_eval, select_cases

CASES = """
- id: scifi-series
  category: entre-dominios
  message: "Quero 3 séries"
  profile:
    liked: ["book:Scifi Book 1", "book:Scifi Book 2"]
  expect:
    domains: [series]
    count: 3
    must_call: [recommend_for_user]
    relevant_tags: [science-fiction]
- id: ambiguous
  category: esclarecimento
  message: "me recomenda algo"
  expect:
    clarification: true
"""


def test_heuristic_eval_end_to_end(project):
    project.cases.mkdir(parents=True, exist_ok=True)
    (project.cases / "test.yaml").write_text(CASES)
    cases = select_cases(project)
    summary = run_llm_eval(project, "heuristic", cases=cases, log=lambda _: None)
    assert summary["cases"] == 2
    assert summary["pass_rate"] == 1.0
    assert summary["graders"]["relevance_tags"] == 1.0
    lines = (project.reports / "llm_heuristic.md").read_text()
    assert "Taxa de aprovação" in lines
    with open(f"{summary['folder']}/results.jsonl") as fh:
        records = [json.loads(line) for line in fh]
    assert records[0]["result"]["tool_calls"][0]["tool"] == "recommend_for_user"
