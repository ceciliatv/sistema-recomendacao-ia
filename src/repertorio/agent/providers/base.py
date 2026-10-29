"""Tipos comuns aos provedores. O laço do agente só conhece estes tipos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Erro de configuração ou de API, com mensagem pronta para mostrar."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema do input


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    invalid_json: str | None = None  # argumentos que não eram JSON válido


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    refused: bool = False


class Conversation(ABC):
    """Uma conversa com histórico no formato nativo do provedor."""

    @abstractmethod
    def add_user(self, text: str) -> None: ...

    @abstractmethod
    def step(self) -> AssistantTurn: ...

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...


class Provider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def start(self, system: str, tools: list[ToolSpec]) -> Conversation: ...
