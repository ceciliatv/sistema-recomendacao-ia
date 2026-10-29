# 001 — Catálogo unificado

**Status:** Em andamento · **Data:** 2026-10-29

## Problema
Livros, filmes, séries e música vêm de fontes com formatos, ids e vocabulários de gênero diferentes. Sem um esquema comum, o motor e o agente precisariam conhecer cada fonte, e a recomendação entre domínios ("li Duna, que série eu veria?") ficaria impossível.

## Objetivo
Um catálogo único, reproduzível a partir de datasets públicos, em que todo item tem o mesmo formato e tags normalizadas num vocabulário compartilhado.

## Proposta
- **Fontes** (baixadas por script, nunca versionadas):
  - Filmes: MovieLens `ml-latest-small` (GroupLens): títulos, gêneros, tags e ~100 mil notas.
  - Livros: Goodbooks-10k (CC BY-SA 4.0): 10 mil livros, ~6 milhões de notas e tags de prateleiras do Goodreads.
  - Música: HetRec 2011 Last.fm 2k (GroupLens): artistas, contagem de execuções por usuário e tags.
  - Séries: API pública do TVMaze (CC BY-SA, sem chave): gêneros, sinopse, nota e popularidade.
- **Esquema do item** (`items.parquet`): `item_id` (`<domínio>:<id na fonte>`), `domain`, `source_id`, `title`, `creators`, `year`, `tags`, `tag_weights`, `description`, `popularity` (percentil no domínio, 0–1), `rating`.
- **Interações** (`interactions_<domínio>.parquet`): `user_id`, `item_id`, `value`, `timestamp`. Só existem para filmes, livros e música; séries não têm interações públicas e são um domínio só de conteúdo (cold start de item por construção).
- **Normalização de tags**: minúsculas, hífens, sinônimos (`sci-fi` → `science-fiction`, `rap` → `hip-hop`), e filtro de tags de prateleira sem significado (`to-read`, `owned`, `seen-live`, `favorites`…). Tags de humor e tema (`dark`, `dystopia`, `epic`, `romance`) são o que liga os domínios.
- **Descartado:** embeddings de texto via API para o catálogo (custo e chave obrigatória para rodar o básico) e APIs ao vivo (TMDB, Spotify), que mudam com o tempo e tornam os números não reproduzíveis.

## Escopo
- **Dentro:** download, limpeza, normalização, escrita em parquet, busca por título.
- **Fora:** atualização incremental, imagens, idiomas além do que as fontes trazem.

## Critérios de aceite
- [ ] `repertorio data download` baixa as quatro fontes para `data/raw/` e é idempotente (não baixa de novo o que já existe).
- [ ] `repertorio data build` gera `data/processed/items.parquet` e `interactions_{book,movie,music}.parquet`.
- [ ] Todo item tem `item_id` único, `domain` válido e ao menos uma tag.
- [ ] As tags dos quatro domínios compartilham vocabulário (há tags que aparecem em pelo menos três domínios).
- [ ] `tests/test_tags.py` cobre sinônimos e o filtro de tags de prateleira.
- [ ] Nenhum dado bruto entra no Git (`data/` no `.gitignore`).

## Clarify
**P1.** Dados públicos ou APIs ao vivo? *Recomendação:* datasets públicos, para que os números do README sejam reproduzíveis.
**Resposta** Datasets públicos. | Cecília, 2026-10-29

**P2.** Usar todos os 53 mil usuários do Goodbooks? *Recomendação:* não; amostrar 10 mil usuários com semente fixa, para treinar em minutos num notebook.
**Decisão** Amostrar 10 mil, semente 42 (padrão adotado na implementação; revisável). | 2026-10-29

## Riscos
- Tags de usuário são ruidosas; o filtro de prateleiras pode deixar passar lixo ou cortar tag útil. Mitigação: lista explícita e testada, mais corte por frequência.
- Licenças: MovieLens e HetRec não permitem redistribuição; por isso o repositório só traz o script de download.
