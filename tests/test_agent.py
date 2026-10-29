import json

import pytest

from repertorio.agent.agent import RecommenderAgent
from repertorio.agent.providers import get_provider
from repertorio.agent.providers.base import AssistantTurn, ProviderError, ToolCall
from repertorio.agent.providers.heuristic import HeuristicProvider
from repertorio.agent.providers.scripted import ScriptedProvider
from repertorio.profile import TasteProfile


def _submit_first_results(results):
    items = json.loads(results[-1][0].content)["items"]
    picks = [{"item_id": i["item_id"], "reason": "porque sim"} for i in items[:2]]
    return AssistantTurn("", [ToolCall("c2", "submit_recommendations", {"recommendations": picks, "message": "ok"})])


def test_loop_runs_tools_and_stops_on_submit(engine):
    provider = ScriptedProvider(
        [
            AssistantTurn("", [ToolCall("c1", "recommend_for_user", {"domain": "movie", "k": 3})]),
            _submit_first_results,
        ]
    )
    agent = RecommenderAgent(provider, engine, TasteProfile(liked=["movie:scifi0"]))
    result = agent.ask("dois filmes")
    assert result.submitted and len(result.recommendations) == 2
    assert result.steps == 2
    assert [c["tool"] for c in result.tool_calls] == ["recommend_for_user", "submit_recommendations"]


def test_hallucinated_ids_are_rejected(engine):
    picks = [{"item_id": "movie:scifi3", "reason": "x"}, {"item_id": "movie:inventado", "reason": "y"}]
    provider = ScriptedProvider(
        [
            AssistantTurn("", [ToolCall("c1", "search_catalog", {"query": "scifi", "domain": "movie"})]),
            AssistantTurn("", [ToolCall("c2", "submit_recommendations", {"recommendations": picks, "message": ""})]),
        ]
    )
    result = RecommenderAgent(provider, engine, TasteProfile()).ask("filmes")
    assert [p["item_id"] for p in result.recommendations] == ["movie:scifi3"]
    assert result.rejected_ids == ["movie:inventado"]


def test_ids_not_returned_by_tools_are_rejected_even_if_they_exist(engine):
    picks = [{"item_id": "book:romance1", "reason": "lembrei"}]
    provider = ScriptedProvider(
        [AssistantTurn("", [ToolCall("c1", "submit_recommendations", {"recommendations": picks, "message": ""})])]
    )
    result = RecommenderAgent(provider, engine, TasteProfile()).ask("um livro")
    assert result.recommendations == [] and result.rejected_ids == ["book:romance1"]


def test_invalid_arguments_return_tool_error_and_loop_continues(engine):
    provider = ScriptedProvider(
        [
            AssistantTurn("", [ToolCall("c1", "recommend_for_user", {"domain": "podcast"})]),
            AssistantTurn("Não consegui, pode reformular?"),
        ]
    )
    result = RecommenderAgent(provider, engine, TasteProfile()).ask("podcasts")
    first = provider.conversation.results[0][0]
    assert first.is_error and "domain" in first.content
    assert result.message == "Não consegui, pode reformular?"
    assert not result.submitted


def test_step_limit(engine):
    loop = [AssistantTurn("", [ToolCall(f"c{i}", "get_user_profile", {})]) for i in range(5)]
    result = RecommenderAgent(ScriptedProvider(loop), engine, TasteProfile(), max_steps=3).ask("oi")
    assert result.error and result.steps == 3


def test_update_profile_resolves_titles(engine):
    provider = ScriptedProvider(
        [
            AssistantTurn(
                "", [ToolCall("c1", "update_profile", {"liked": ["book:Horror Book 2", "book:nao existe zz"]})]
            ),
            AssistantTurn("Anotado!"),
        ]
    )
    agent = RecommenderAgent(provider, engine, TasteProfile())
    agent.ask("amei Horror Book 2")
    assert agent.profile.liked == ["book:horror2"]


def test_heuristic_provider_end_to_end(engine):
    agent = RecommenderAgent(HeuristicProvider(), engine, TasteProfile(liked=["book:scifi0"]))
    result = agent.ask("Me recomenda 3 séries sem terror")
    assert result.submitted and len(result.recommendations) == 3
    assert all(p["domain"] == "series" for p in result.recommendations)


def test_heuristic_asks_when_domain_is_unknown(engine):
    result = RecommenderAgent(HeuristicProvider(), engine, TasteProfile()).ask("me recomenda algo")
    assert not result.submitted and result.message.endswith("?")


def test_unknown_provider_has_clear_error():
    with pytest.raises(ProviderError):
        get_provider("gemini")
