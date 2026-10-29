# Avaliação offline — protocolo `loo`

## book

9991 usuários, 9989 itens, 768250 positivos; 9991 usuários avaliados.

| Modelo | HR@10 | Recall@20 | NDCG@10 | Cobertura@10 | Treino (s) |
|---|---:|---:|---:|---:|---:|
| popularity | 0.0469 | 0.0781 | 0.0273 | 0.0045 | 0.0 |
| itemknn | 0.1548 | 0.2126 | 0.0936 | 0.2049 | 1.0 |
| two_tower | **0.1739** | **0.2423** | **0.1044** | **0.5958** | 230.4 |

## movie

601 usuários, 2808 itens, 44149 positivos; 601 usuários avaliados.

| Modelo | HR@10 | Recall@20 | NDCG@10 | Cobertura@10 | Treino (s) |
|---|---:|---:|---:|---:|---:|
| popularity | 0.0399 | 0.0699 | 0.0218 | 0.0217 | 0.0 |
| itemknn | 0.0599 | 0.1015 | 0.0271 | 0.0716 | 0.1 |
| two_tower | **0.0632** | **0.1148** | **0.0313** | **0.3939** | 26.6 |

## music

1868 usuários, 4442 itens, 76843 positivos; 1868 usuários avaliados.

| Modelo | HR@10 | Recall@20 | NDCG@10 | Cobertura@10 | Treino (s) |
|---|---:|---:|---:|---:|---:|
| popularity | 0.0862 | 0.1204 | 0.0468 | 0.0068 | 0.0 |
| itemknn | 0.2238 | 0.3067 | **0.1407** | 0.1817 | 0.1 |
| two_tower | **0.2377** | **0.3137** | 0.1397 | **0.4692** | 39.8 |
