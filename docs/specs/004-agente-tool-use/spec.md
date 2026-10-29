# 004 — Agente com tool use (Anthropic e OpenAI)

**Status:** Concluída · **Data:** 2026-10-29

## Problema
O motor da spec 003 responde a chamadas estruturadas, mas as pessoas pedem em linguagem natural: "quero um livro curto parecido com Black Mirror, nada de terror". Um LLM sozinho entende o pedido, mas inventa títulos e ignora o catálogo. Precisamos do melhor dos dois.

## Objetivo
Um agente que interpreta o pedido, consulta o catálogo e os modelos **por ferramentas** e devolve recomendações estruturadas, ancoradas em itens reais, com Claude ou GPT trocáveis por configuração.

## Proposta
- **Ferramentas** (JSON Schema, validadas com Pydantic antes de executar; erro de validação volta como `tool_result` de erro para o modelo corrigir):
  - `search_catalog(query, domain?, year_min?, year_max?, exclude_tags?, limit?)`
  - `get_item_details(item_id)`
  - `get_user_profile()`
  - `recommend_for_user(domain, k?, year_min?, year_max?, exclude_tags?, require_tags?)`
  - `find_similar(item_id, target_domain?, k?, year_min?, year_max?, exclude_tags?)`
  - `update_profile(liked?, disliked?, preferred_tags?, avoided_tags?)`
  - `submit_recommendations(recommendations: [{item_id, reason}], message)`: ferramenta terminal; é por ela que a resposta sai estruturada.
- **Laço do agente** independente de provedor: `Conversation.step()` devolve texto e chamadas de ferramenta normalizados; o laço executa as ferramentas, devolve todos os resultados de uma vez e para em `submit_recommendations`, em texto puro (pergunta de esclarecimento) ou no limite de passos.
- **Provedores:**
  - `AnthropicProvider` (SDK `anthropic`): `claude-opus-5-5` por padrão, `effort` explícito, cache de prompt automático, fallback de recusa do lado do servidor (`fallbacks: "default"`), tratamento de `stop_reason == "refusal"` e o conteúdo da resposta devolvido inteiro ao histórico (inclui blocos de thinking).
  - `OpenAIProvider` (SDK `openai`, Chat Completions com function calling).
  - `HeuristicProvider`: agente por regras, sem LLM, que segue o mesmo protocolo. É a linha de base dos evals e permite rodar tudo sem chave.
  - `ScriptedProvider`: respostas roteirizadas para os testes.
- **Ancoragem (grounding):** o `ToolBox` registra todo `item_id` que apareceu em resultado de ferramenta. Recomendação com id fora desse conjunto é descartada e contada como alucinação.
- **Segurança:** o prompt de sistema trata resultados de ferramenta e o texto do usuário como dados; pedidos para ignorar regras ou recomendar itens fora do catálogo não mudam o protocolo.
- **Escolha forçada de ferramenta:** não usamos `tool_choice` forçado (os modelos atuais da Anthropic não aceitam `any`/`tool`); o prompt pede o `submit_recommendations` e o laço confere.

## Escopo
- **Dentro:** ferramentas, laço, quatro provedores, prompts, `repertorio chat`.
- **Fora:** streaming de tokens na CLI, memória entre sessões além do JSON do perfil.

## Critérios de aceite
- [x] Com o `ScriptedProvider`, o laço executa ferramentas, devolve resultados e termina no `submit_recommendations` (`tests/test_agent.py`).
- [x] Ids inventados no `submit_recommendations` são descartados e marcados como não ancorados.
- [x] Argumentos inválidos geram `tool_result` de erro, sem derrubar o laço.
- [x] O `HeuristicProvider` responde aos casos de eval sem chave de API.
- [x] Sem chave configurada, `repertorio chat --provider anthropic` falha com mensagem clara, sem stack trace.

## Clarify
**P1.** Há chaves de API para validar os provedores reais agora? *Recomendação:* não bloquear: implementar os dois provedores conforme a documentação dos SDKs, testar o laço com o provedor roteirizado e rodar os evals reais quando as chaves existirem.
**Resposta** Ainda não há chaves. | Cecília, 2026-10-29

**P2.** Usar o Tool Runner do SDK da Anthropic? *Recomendação:* não; o laço precisa ser o mesmo para Claude, GPT e a linha de base, senão os evals comparam implementações diferentes.
**Decisão** Laço manual, comum aos provedores. | 2026-10-29

## Riscos
- Os provedores reais só foram verificados contra a assinatura dos SDKs instalados, não contra a API. O primeiro `repertorio eval llm` com chave é o teste de fumaça.
- Modelos podem responder em texto em vez de chamar `submit_recommendations`; o grader `structured` mede isso.
