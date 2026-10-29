"""GPT via SDK oficial da OpenAI (Chat Completions com function calling)."""

from __future__ import annotations

import json

import openai

from repertorio.agent.providers.base import (
    AssistantTurn,
    Conversation,
    Provider,
    ProviderError,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


class OpenAIConversation(Conversation):
    def __init__(
        self, client: openai.OpenAI, model: str, reasoning_effort: str | None, system: str, tools: list[ToolSpec]
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> AssistantTurn:
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_completion_tokens=16000,
                **extra,
            )
        except openai.AuthenticationError as exc:
            raise ProviderError("OpenAI: credencial inválida ou ausente (defina OPENAI_API_KEY).") from exc
        except openai.NotFoundError as exc:
            raise ProviderError(f"OpenAI: modelo {self.model!r} não encontrado.") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError("OpenAI: sem conexão com a API.") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI: erro {exc.status_code}: {exc.message}") from exc

        choice = response.choices[0]
        message = choice.message
        details = getattr(response.usage, "prompt_tokens_details", None)
        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        )
        if getattr(message, "refusal", None):
            return AssistantTurn(text=message.refusal, stop_reason="refusal", usage=usage, refused=True)

        assistant: dict = {"role": "assistant", "content": message.content or ""}
        calls = []
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
            for tc in message.tool_calls:
                try:
                    calls.append(ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}")))
                except json.JSONDecodeError:
                    calls.append(ToolCall(tc.id, tc.function.name, {}, invalid_json=tc.function.arguments))
        self.messages.append(assistant)
        return AssistantTurn(
            text=message.content or "", tool_calls=calls, stop_reason=choice.finish_reason or "", usage=usage
        )

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            content = f"ERRO: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str, reasoning_effort: str | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort
        try:
            self.client = openai.OpenAI()
        except openai.OpenAIError as exc:
            raise ProviderError("OpenAI: defina OPENAI_API_KEY no ambiente ou no .env.") from exc

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        return OpenAIConversation(self.client, self.model, self.reasoning_effort, system, tools)
