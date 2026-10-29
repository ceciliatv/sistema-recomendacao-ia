# 003 — Motor de recomendação, perfil e cold start

**Status:** Concluída · **Data:** 2026-10-29

## Problema
Os modelos da spec 002 só sabem ranquear dentro de um domínio e só quando a pessoa já tem histórico nele. Quem chega dizendo "amei o livro Duna" e pede uma série cai em dois buracos: não há histórico de séries (cold start de usuário no domínio) e séries não têm interações (cold start de item).

## Objetivo
Um motor que, a partir de um perfil de gosto com itens de qualquer domínio, recomenda em qualquer domínio, respeita filtros e explica cada sugestão.

## Proposta
- **Perfil de gosto** (`TasteProfile`): itens curtidos, itens rejeitados, tags preferidas e tags a evitar. Serializável em JSON.
- **Índice de conteúdo**: vetores TF-IDF esparsos sobre as tags normalizadas (com peso por tag). Esparso de propósito: as tags em comum entre o perfil e o item viram a explicação ("tem dystopia, dark e science-fiction, como os itens que você curtiu").
- **Busca textual**: TF-IDF sobre título, criadores, tags e descrição, com filtros e um leve reforço de popularidade.
- **Mistura por domínio-alvo** (scores convertidos em percentil antes de somar, para serem comparáveis):
  - com histórico no domínio e modelo treinado: `score = a·colaborativo + (1−a)·conteúdo + 0,05·popularidade`, com `a = n/(n+3)` (`n` = itens curtidos no domínio). Pouco histórico pesa mais o conteúdo; muito histórico pesa mais o colaborativo;
  - sem histórico no domínio (ou domínio sem interações, como séries): conteúdo a partir do perfil inteiro, entre domínios;
  - perfil vazio: popularidade, filtrada pelas tags preferidas quando houver.
- **Filtros**: ano mínimo e máximo, tags proibidas, tags exigidas, itens já consumidos sempre fora.
- **Explicações** por item: o item curtido mais parecido (pelos embeddings do Two-Tower ou pelo ItemKNN), as tags em comum e a origem do score.
- **Similares** (`similar_items`): no mesmo domínio, mistura de embeddings e conteúdo; entre domínios, só conteúdo.

## Escopo
- **Dentro:** `profile.py`, `retrieval/content.py`, `engine.py` e os testes.
- **Fora:** feedback implícito em tempo real e re-treino online.

## Critérios de aceite
- [x] Perfil só com livros recebe recomendações de séries (cross-domain) com tags em comum na explicação.
- [x] Perfil vazio recebe itens populares; com `preferred_tags`, só itens com essas tags.
- [x] Nenhuma recomendação viola filtros nem repete item do perfil (`tests/test_engine.py`).
- [x] O peso do colaborativo cresce com o histórico no domínio (teste do `a`).
- [x] `repertorio recommend --like "book:Dune" --domain series` imprime itens com explicação.

## Clarify
**P1.** Embeddings densos (via API ou sentence-transformers) para o conteúdo? *Recomendação:* não por agora. TF-IDF de tags roda offline, sem chave, e a sobreposição de tags é a própria explicação. Embeddings densos podem entrar depois atrás da mesma interface.
**Decisão** TF-IDF de tags. | 2026-10-29

## Riscos
- Música tem tags de gênero musical que não existem em livros e filmes; a ponte entre domínios depende de tags de humor (`dark`, `melancholic`, `epic`). Quando a ponte é fraca, o agente compensa buscando por humor com `search_catalog`.
