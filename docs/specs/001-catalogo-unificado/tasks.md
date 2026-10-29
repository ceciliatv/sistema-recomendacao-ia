# 001 — Tasks

- [x] T1. `data/schema.py`: domínios, colunas do item e das interações.
- [x] T2. `data/tags.py`: normalização, sinônimos, expansões e filtro de prateleiras (+ testes).
- [x] T3. `data/download.py`: download idempotente das quatro fontes (TVMaze paginado, com pausa entre páginas).
- [x] T4. `data/build.py`: leitores por fonte → esquema único → parquet; popularidade em percentil por domínio.
- [x] T5. `catalog.py`: carregar o parquet, buscar por id e por título (com e sem domínio).
- [x] T6. Conferir o vocabulário compartilhado entre domínios (critério 4).
