"""Laço do agente, igual para todos os provedores."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from repertorio.agent.prompts import SYSTEM_PROMPT
from repertorio.agent.providers.base import Provider, Usage
from repertorio.agent.tools import ToolBox
from repertorio.engine import RecommendationEngine
from repertorio.profile import TasteProfile


@dataclass
class AgentResult:
    message: str
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    submitted: bool = False
    refused: bool = False
    error: str | None = None
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


class RecommenderAgent:
    def __init__(self, provider: Provider, engine: RecommendationEngine, profile: TasteProfile, max_steps: int = 8):
        self.provider = provider
        self.toolbox = ToolBox(engine, profile)
        self.conversation = provider.start(SYSTEM_PROMPT, self.toolbox.specs)
        self.max_steps = max_steps

    @property
    def profile(self) -> TasteProfile:
        return self.toolbox.profile

    def ask(self, message: str) -> AgentResult:
        started = time.perf_counter()
        self.toolbox.new_turn()
        self.conversation.add_user(message)
        result = AgentResult(message="")
        for step in range(1, self.max_steps + 1):
            turn = self.conversation.step()
            result.steps = step
            result.usage.add(turn.usage)
            if turn.refused:
                result.refused = True
                result.message = turn.text
                break
            if not turn.tool_calls:
                result.message = turn.text
                break
            outputs = [self.toolbox.execute(call) for call in turn.tool_calls]
            self.conversation.add_tool_results(outputs)
            submission = self.toolbox.submission
            if submission is not None:
                result.submitted = True
                result.recommendations = submission.picks
                result.rejected_ids = submission.rejected
                result.message = submission.message or turn.text
                break
        else:
            result.error = f"limite de {self.max_steps} passos sem resposta final"
        result.tool_calls = list(self.toolbox.trace)
        result.latency_s = round(time.perf_counter() - started, 3)
        return result
