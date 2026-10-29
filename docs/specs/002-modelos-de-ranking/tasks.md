# 002 — Tasks

- [ ] T1. `models/base.py`: `InteractionData` (matriz esparsa + índices), split `loo` e `new-users`, filtro k-core.
- [ ] T2. `evals/metrics.py`: HR, Recall, NDCG, cobertura (+ testes calculados à mão).
- [ ] T3. `models/popularity.py` e `models/itemknn.py` (similaridade em blocos, poda top-k).
- [ ] T4. `models/two_tower.py`: módulo PyTorch, treino com softmax completo e histórico de tamanho aleatório, save/load.
- [ ] T5. `models/store.py`: salvar e carregar os modelos treinados por domínio.
- [ ] T6. `evals/offline.py`: roda os protocolos e escreve o relatório.
- [ ] T7. Teste com dados sintéticos em que os modelos personalizados vencem a popularidade.
