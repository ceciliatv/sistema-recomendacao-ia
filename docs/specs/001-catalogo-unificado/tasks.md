# 001 — Tasks

- [ ] T1. `data/schema.py`: domínios, colunas do item e das interações.
- [ ] T2. `data/tags.py`: normalização, sinônimos, expansões e filtro de prateleiras (+ testes).
- [ ] T3. `data/download.py`: download idempotente das quatro fontes (TVMaze paginado, com pausa entre páginas).
- [ ] T4. `data/build.py`: leitores por fonte → esquema único → parquet; popularidade em percentil por domínio.
- [ ] T5. `catalog.py`: carregar o parquet, buscar por id e por título (com e sem domínio).
- [ ] T6. Conferir o vocabulário compartilhado entre domínios (critério 4).
