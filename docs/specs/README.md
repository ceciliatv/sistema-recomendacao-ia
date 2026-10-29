# Specs do Repertório

Este projeto foi construído com **spec-driven development**: cada parte do sistema começou como uma spec curta, com problema, proposta e critérios de aceite checáveis, e só depois virou código. O histórico do Git segue essa ordem (`docs(specs)` primeiro, `feat(NNN)` depois).

## Fluxo

1. **Spec** (`NNN-nome/spec.md`, a partir do `_template`): problema, objetivo, proposta, escopo, critérios de aceite, perguntas em aberto e riscos.
2. **Clarify**: cada pergunta em aberto ganha uma recomendação e uma resposta registrada ("**Resposta** … | nome, data"). A proposta e os critérios são ajustados à resposta.
3. **Tasks** (`NNN-nome/tasks.md`): a lista de tarefas, em ordem, com o critério que cada uma fecha.
4. **Implementação**: um commit por spec, com o prefixo `feat(NNN)`.
5. **Verificação**: testes, lint e os comandos citados nos critérios. Só é marcado `[x]` o que foi verificado de fato.

## Índice

| Spec | Assunto | Status |
|---|---|---|
| [001](001-catalogo-unificado/spec.md) | Catálogo unificado (livros, filmes, séries, música) | Concluída |
| [002](002-modelos-de-ranking/spec.md) | Modelos de ranking e avaliação offline (PyTorch) | Concluída |
| [003](003-motor-e-cold-start/spec.md) | Motor de recomendação, perfil e cold start | Concluída |
| [004](004-agente-tool-use/spec.md) | Agente com tool use (Anthropic e OpenAI) | Concluída |
| [005](005-evals-llm/spec.md) | Evals das respostas do agente | Concluída |
| [006](006-interfaces-e-ci/spec.md) | CLI, API e CI | Concluída |
