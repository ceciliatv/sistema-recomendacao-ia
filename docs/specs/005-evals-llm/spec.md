# 005 — Evals das respostas do agente

**Status:** Concluída · **Data:** 2026-10-29

## Problema
Métricas offline (spec 002) medem o ranking, não a resposta que a pessoa recebe. Trocar de modelo, de prompt ou de ferramenta muda a qualidade de um jeito que só se vê olhando muitas conversas. Sem evals, cada mudança é no escuro.

## Objetivo
Um conjunto de casos versionado e um runner que mede, por provedor e por categoria, se o agente responde certo, de forma ancorada e respeitando restrições, com graders determinísticos e um juiz LLM opcional.

## Proposta
- **Casos** (`evals/cases/*.yaml`): mensagem, perfil (por título, resolvido no catálogo ao carregar) e expectativas: domínios, quantidade, ano, tags proibidas, ferramentas esperadas, tags relevantes, se deve pedir esclarecimento.
- **Categorias:** personalização num domínio, entre domínios, restrições, cold start, esclarecimento e robustez (injeção de prompt, item inexistente).
- **Graders determinísticos** (0–1 por caso):
  - `structured`: respondeu pelo `submit_recommendations`;
  - `grounded`: fração dos ids que existem e vieram de ferramenta;
  - `domain`: fração no domínio pedido;
  - `count`: quantidade dentro do intervalo;
  - `constraints`: fração que respeita ano, tags proibidas e itens já consumidos;
  - `tool_use`: chamou alguma das ferramentas esperadas;
  - `relevance_tags`: fração com ao menos uma tag relevante (proxy barato de relevância);
  - `clarification`: quando o caso pede, perguntou em vez de recomendar.
- **Aprovação do caso:** todos os graders de checagem aplicáveis em 1,0 (`structured`, `grounded`, `domain`, `count`, `constraints`, `tool_use`, `clarification`, `safety`) **e** `relevance_tags` ≥ 0,5.
- **Juiz LLM** (opcional, `--judge`): nota de 1 a 5 para relevância, personalização e qualidade da explicação, devolvida por ferramenta. Por padrão o juiz é do **outro** provedor (Claude julga GPT e vice-versa), para reduzir o viés de autoavaliação.
- **Saída:** `reports/llm/<run_id>/results.jsonl` (um registro por caso, com rastro de ferramentas) e `summary.md` (média por grader e por categoria, latência e tokens).
- **Custo:** o runner mostra quantas chamadas vai fazer e pede confirmação (`--yes` pula), porque cada execução com provedor real gasta dinheiro.

## Escopo
- **Dentro:** casos, graders, juiz, runner, relatório, linha de base heurística.
- **Fora:** dashboards, comparação estatística automática entre execuções.

## Critérios de aceite
- [x] Pelo menos 24 casos cobrindo as seis categorias.
- [x] `repertorio eval llm --provider heuristic --yes` roda sem chave e gera o relatório.
- [x] Cada grader tem teste com caso que passa e caso que falha (`tests/test_graders.py`).
- [x] O juiz é opcional e, quando ligado, usa o provedor oposto por padrão.
- [x] O README mostra a linha de base heurística e deixa claro que as linhas de Claude e GPT só entram com números medidos.

## Clarify
**P1.** Sem chaves agora, o que publicar no README? *Recomendação:* só números medidos: a linha de base heurística e as métricas offline. As linhas dos LLMs ficam vazias, com o comando para gerá-las.
**Resposta** Sem chaves por enquanto. | Cecília, 2026-10-29

## Riscos
- `relevance_tags` é um proxy grosseiro; por isso existe o juiz.
- Juiz LLM tem viés de tamanho e de estilo; a rubrica pede notas por critério e justificativa curta.

## Registro de implementação
- 2026-10-29: a primeira rodada da linha de base heurística aprovou 92,6% dos casos, inclusive "séries documentais" → Breaking Bad. A aprovação ignorava relevância; passou a exigir `relevance_tags` ≥ 0,5.
- 2026-10-29: o caso `rob-prompt-leak` revelou que a busca por "Radiohead" devolvia LazyTown (tag de troll `better-than-radiohead` num documento curto). A busca passou a somar a similaridade de título.
