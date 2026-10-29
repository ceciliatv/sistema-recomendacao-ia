from repertorio.agent.agent import AgentResult
from repertorio.agent.providers.base import AssistantTurn, ToolCall
from repertorio.agent.providers.scripted import ScriptedProvider
from repertorio.evals.llm.cases import EvalCase, Expectation
from repertorio.evals.llm.judge import judge, opposite
from repertorio.profile import TasteProfile

CASE = EvalCase(id="t", category="personalizacao", message="2 filmes", profile={}, expect=Expectation())
RESULT = AgentResult(
    message="ok",
    submitted=True,
    recommendations=[{"item_id": "movie:scifi1", "reason": "r", "title": "x", "domain": "movie"}],
)


def test_judge_defaults_to_the_other_provider():
    assert opposite("anthropic") == "openai"
    assert opposite("openai") == "anthropic"


def test_judge_normalizes_grades(catalog):
    grade = {"relevance": 5, "personalization": 3, "explanation": 1, "rationale": "ok"}
    provider = ScriptedProvider([AssistantTurn("", [ToolCall("g", "submit_grade", grade)])])
    scores = judge(provider, CASE, RESULT, catalog, TasteProfile(liked=["movie:scifi0"]))
    assert scores["judge_relevance"] == 1.0
    assert scores["judge_personalization"] == 0.5
    assert scores["judge_explanation"] == 0.0
    assert "movie:scifi1" not in provider.conversation.user_messages[0]  # o juiz vê títulos, não ids


def test_judge_returns_none_on_invalid_grade(catalog):
    provider = ScriptedProvider([AssistantTurn("", [ToolCall("g", "submit_grade", {"relevance": 9})])])
    assert judge(provider, CASE, RESULT, catalog, TasteProfile()) is None


def test_judge_skips_answers_without_recommendations(catalog):
    empty = AgentResult(message="?")
    assert judge(ScriptedProvider([]), CASE, empty, catalog, TasteProfile()) is None
