"""Provedor roteirizado para testes: devolve turnos pré-definidos e guarda o que recebeu."""

from __future__ import annotations

from collections.abc import Callable

from repertorio.agent.providers.base import AssistantTurn, Conversation, Provider, ToolResult, ToolSpec

# um passo pode ser um turno pronto ou uma função que olha os resultados anteriores
Step = AssistantTurn | Callable[[list[list[ToolResult]]], AssistantTurn]


class ScriptedConversation(Conversation):
    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.user_messages: list[str] = []
        self.results: list[list[ToolResult]] = []

    def add_user(self, text: str) -> None:
        self.user_messages.append(text)

    def step(self) -> AssistantTurn:
        if not self.steps:
            return AssistantTurn(text="(fim do roteiro)")
        step = self.steps.pop(0)
        return step(self.results) if callable(step) else step

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.results.append(results)


class ScriptedProvider(Provider):
    name = "scripted"
    model = "script"

    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.conversation: ScriptedConversation | None = None

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        self.conversation = ScriptedConversation(self.steps)
        return self.conversation
