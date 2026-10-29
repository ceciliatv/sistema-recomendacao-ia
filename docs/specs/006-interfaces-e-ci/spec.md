# 006 — CLI, API e CI

**Status:** Em andamento · **Data:** 2026-10-29

## Problema
Sem uma porta de entrada única, rodar o projeto exige conhecer a ordem dos scripts. Sem CI, nada garante que o `main` continua passando.

## Objetivo
Uma CLI que cobre o ciclo inteiro (dados → treino → avaliação → uso), uma API HTTP para integrar com outros apps e um CI que roda lint e testes sem rede e sem chave.

## Proposta
- **CLI** (Typer + Rich), comando `repertorio`:
  - `data download`, `data build`, `train`;
  - `eval offline`, `eval llm`;
  - `search`, `recommend`, `chat`, `serve`.
- **API** (FastAPI): `GET /health`, `GET /items/search`, `GET /items/{id}`, `GET /items/{id}/similar`, `POST /recommendations`, `POST /chat`. Os artefatos carregam uma vez no `lifespan`.
- **Configuração** por variáveis de ambiente (`.env`): chaves e modelos. Nenhuma chave no código.
- **CI** (GitHub Actions): `ruff check` e `pytest` em Python 3.12 e 3.13, com PyTorch CPU. Os testes usam um catálogo sintético e não acessam a rede.

## Critérios de aceite
- [ ] `repertorio --help` lista os comandos acima.
- [ ] `tests/test_api.py` cobre os endpoints com o catálogo sintético.
- [ ] `ruff check .` e `pytest` passam localmente.
- [ ] O workflow `.github/workflows/ci.yml` passa no GitHub.

## Riscos
- PyTorch deixa o CI mais lento; o índice CPU do PyTorch reduz o download.
