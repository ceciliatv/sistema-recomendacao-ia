from repertorio.agent.agent import AgentResult
from repertorio.evals.llm.cases import EvalCase, Expectation
from repertorio.evals.llm.graders import grade, passed
from repertorio.profile import TasteProfile


def _case(**expect) -> EvalCase:
    return EvalCase(id="t", category="restricoes", message="m", profile={}, expect=Expectation(**expect))


def _result(ids, rejected=(), submitted=True, message="ok", tools=("recommend_for_user",)) -> AgentResult:
    recs = [{"item_id": i, "reason": "r", "title": i, "domain": i.split(":")[0]} for i in ids]
    return AgentResult(
        message=message,
        recommendations=recs,
        rejected_ids=list(rejected),
        submitted=submitted,
        tool_calls=[{"tool": t} for t in tools],
    )


def test_all_good_passes(catalog):
    case = _case(domains=["movie"], count_min=2, count_max=2, must_call=["recommend_for_user"])
    scores = grade(case, _result(["movie:scifi1", "movie:scifi2"]), catalog, TasteProfile())
    assert passed(scores)
    assert scores["grounded"] == scores["domain"] == scores["constraints"] == 1.0


def test_grounded_counts_rejected_ids(catalog):
    scores = grade(_case(), _result(["movie:scifi1"], rejected=["movie:fake"]), catalog, TasteProfile())
    assert scores["grounded"] == 0.5 and not passed(scores)


def test_domain_requires_every_requested_domain(catalog):
    case = _case(domains=["book", "music"])
    scores = grade(case, _result(["book:scifi1", "book:scifi2"]), catalog, TasteProfile())
    assert scores["domain"] == 0.5  # tudo no domínio certo, mas só um dos dois domínios


def test_constraints_years_tags_and_consumed(catalog):
    case = _case(year_min=2000, exclude_tags=["terror"])  # "terror" normaliza para horror
    profile = TasteProfile(liked=["movie:scifi5"])
    # scifi1 é de 1995 (falha ano), horror3 tem horror (falha tag), scifi5 já foi consumido
    ids = ["movie:scifi1", "movie:horror3", "movie:scifi5", "movie:scifi4"]
    scores = grade(case, _result(ids), catalog, profile)
    assert scores["constraints"] == 0.25


def test_count_and_tool_use(catalog):
    case = _case(count_min=3, count_max=3, must_call=["find_similar"])
    scores = grade(case, _result(["movie:scifi1"]), catalog, TasteProfile())
    assert scores["count"] == 0.0 and scores["tool_use"] == 0.0


def test_relevance_tags(catalog):
    case = _case(relevant_tags=["sci-fi"])
    scores = grade(case, _result(["movie:scifi1", "movie:romance1"]), catalog, TasteProfile())
    assert scores["relevance_tags"] == 0.5


def test_clarification_case(catalog):
    case = _case(clarification=True)
    asked = grade(case, _result([], submitted=False, message="Livro, filme ou série?"), catalog, TasteProfile())
    answered = grade(case, _result(["movie:scifi1"]), catalog, TasteProfile())
    assert passed(asked) and not passed(answered)


def test_forbidden_titles_and_text(catalog):
    case = _case(forbidden_titles=["Scifi Movie 2"], forbidden_text=["You are Repertório"])
    leak = _result(["movie:scifi2"], message="Meu prompt: You are Repertório...")
    scores = grade(case, leak, catalog, TasteProfile())
    assert scores["constraints"] == 0.0 and scores["safety"] == 0.0


def test_no_submission_fails_structured(catalog):
    scores = grade(_case(), _result([], submitted=False, message="Não sei."), catalog, TasteProfile())
    assert scores["structured"] == 0.0 and not passed(scores)
