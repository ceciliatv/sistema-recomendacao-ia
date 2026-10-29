# 002 — Modelos de ranking e avaliação offline

**Status:** Concluída · **Data:** 2026-10-29

## Problema
O agente precisa de um sinal de personalização que não dependa do LLM "lembrar" de itens: dado o que a pessoa gostou, quais itens do catálogo têm mais chance de agradar? Sem uma avaliação offline com protocolo fixo, não há como saber se um modelo é melhor que "recomendar o mais popular".

## Objetivo
Três modelos comparáveis por domínio, com interface única, e um relatório offline reproduzível com métricas de ranking.

## Proposta
- **Interface** `Recommender`: `fit(train)` e `score(histories) -> matriz (usuários × itens)`. Todos os modelos recebem o **histórico** (lista de itens) em vez de um id de usuário. Isso torna os três *indutivos*: funcionam para quem não estava no treino, que é o caso da pessoa conversando com o agente.
- **Popularidade**: contagem de interações positivas. Linha de base obrigatória.
- **ItemKNN**: similaridade de cosseno item-item sobre a matriz binária, com poda top-k por item. Clássico, forte e explicável.
- **Two-Tower com pooling de histórico (PyTorch)**: a torre do usuário é a média dos embeddings dos itens do histórico, mais um MLP residual; a torre do item é uma tabela de embeddings com viés. Treino com softmax completo sobre o catálogo (até ~10 mil itens cabe com folga), mascarando os outros itens do histórico, e com histórico de tamanho aleatório para o modelo aprender a lidar com perfis curtos.
- **Descartado:** matrix factorization com embedding por usuário (transdutiva, exigiria re-treino para cada pessoa nova) e EASE (matriz densa itens × itens grande demais para os 17 mil artistas do Last.fm).
- **Positivos:** nota ≥ 4 (filmes, livros); qualquer execução (música). Filtro k-core: usuários com ≥ 5 positivos, itens com ≥ 3.
- **Protocolos:**
  - `loo` (leave-one-out): último positivo de cada usuário vai para o teste (por timestamp no MovieLens; sorteado com semente nos outros).
  - `new-users`: 20% dos usuários ficam fora do treino; metade do histórico deles vira entrada e a outra metade é o alvo. Mede a capacidade indutiva.
- **Métricas** (ranking completo, sem amostragem de negativos, itens do histórico mascarados): HR@10, Recall@20, NDCG@10 e cobertura do catálogo@10.

## Escopo
- **Dentro:** os três modelos, salvar e carregar artefatos, relatório em Markdown e JSON.
- **Fora:** busca de hiperparâmetros automatizada, GPU obrigatória.

## Critérios de aceite
- [x] `repertorio train` treina e salva os modelos de filmes, livros e música em `artifacts/`.
- [x] `repertorio eval offline --protocol loo` e `--protocol new-users` geram `reports/offline_<protocolo>.md` e `.json`.
- [x] Métricas testadas com casos calculados à mão (`tests/test_metrics.py`).
- [x] Em dados sintéticos com grupos de gosto, ItemKNN e Two-Tower superam a popularidade (`tests/test_models.py`).
- [x] Sementes fixas: duas execuções do mesmo protocolo dão os mesmos números de Popularidade e ItemKNN.

## Clarify
**P1.** Avaliar com negativos amostrados (100 negativos por positivo, comum em artigos) ou ranking completo? *Recomendação:* ranking completo; amostragem infla as métricas e muda a ordem entre modelos.
**Decisão** Ranking completo. | 2026-10-29

**P2.** Ajustar hiperparâmetros no conjunto de teste? *Recomendação:* não. Os valores padrão ficam fixos e documentados; nenhum ajuste usa o teste.
**Decisão** Hiperparâmetros fixos no `ModelConfig`. | 2026-10-29

## Riscos
- O Two-Tower pode perder do ItemKNN em dados pequenos (MovieLens small). Isso é resultado, não bug: o relatório mostra o que der.
- Softmax completo escala com o número de itens; acima de ~100 mil itens seria preciso amostrar negativos.

## Registro de implementação
- 2026-10-29: o modelo servido por domínio (`serving_model`) foi escolhido pelo NDCG@10 no protocolo `new-users`: Two-Tower em livros, ItemKNN em filmes e música. É uma escolha entre dois candidatos com hiperparâmetros fixos, então o viés de selecionar no próprio relatório é pequeno, mas existe: os números dos modelos escolhidos são levemente otimistas.
