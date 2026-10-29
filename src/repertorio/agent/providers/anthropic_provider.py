"""Claude via SDK oficial da Anthropic (Messages API com tool use)."""

from __future__ import annotations

import anthropic

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

# fallback do lado do servidor: se um classificador recusar o pedido, a API refaz a
# chamada num modelo recomendado para aquela categoria, dentro da mesma requisição
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicConversation(Conversation):
    def __init__(self, client: anthropic.Anthropic, model: str, effort: str, system: str, tools: list[ToolSpec]):
        self.client = client
        self.model = model
        self.effort = effort
        self.system = system
        # ordem fixa das ferramentas: mantém o prefixo estável para o cache de prompt
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> AssistantTurn:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Anthropic: credencial inválida ou ausente (defina ANTHROPIC_API_KEY).") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Anthropic: modelo {self.model!r} não encontrado.") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Anthropic: sem conexão com a API.") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic: erro {exc.status_code}: {exc.message}") from exc
        except TypeError as exc:
            # sem nenhuma credencial, o SDK falha ao montar os headers, antes da requisição
            if "authentication" not in str(exc).lower():
                raise
            raise ProviderError(
                "Anthropic: nenhuma credencial encontrada (defina ANTHROPIC_API_KEY no .env ou rode `ant auth login`)."
            ) from exc

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        if response.stop_reason == "refusal":
            return AssistantTurn(text="", stop_reason="refusal", usage=usage, refused=True)

        # o conteúdo volta inteiro para o histórico (inclui blocos de thinking, que
        # precisam ser devolvidos sem alteração)
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        return AssistantTurn(text=text, tool_calls=calls, stop_reason=response.stop_reason or "", usage=usage)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # todos os resultados numa única mensagem, como a API espera para chamadas paralelas
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                    for r in results
                ],
            }
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, effort: str = "medium"):
        self.model = model
        self.effort = effort
        try:
            self.client = anthropic.Anthropic()
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"Anthropic: não foi possível criar o cliente ({exc}).") from exc

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        return AnthropicConversation(self.client, self.model, self.effort, system, tools)
