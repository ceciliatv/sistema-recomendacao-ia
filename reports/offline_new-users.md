# Avaliação offline — protocolo `new-users`

## book

9991 usuários, 9989 itens, 768250 positivos; 1998 usuários avaliados.

| Modelo | HR@10 | Recall@20 | NDCG@10 | Cobertura@10 | Treino (s) |
|---|---:|---:|---:|---:|---:|
| popularity | 0.5916 | 0.1402 | 0.1871 | 0.0025 | 0.0 |
| itemknn | 0.9399 | 0.3236 | 0.4352 | 0.1307 | 0.9 |
| two_tower | **0.9600** | **0.3588** | **0.4812** | **0.2807** | 184.3 |

## movie

601 usuários, 2808 itens, 44149 positivos; 120 usuários avaliados.

| Modelo | HR@10 | Recall@20 | NDCG@10 | Cobertura@10 | Treino (s) |
|---|---:|---:|---:|---:|---:|
| popularity | 0.7250 | 0.2062 | 0.2337 | 0.0078 | 0.0 |
| itemknn | **0.8583** | **0.2993** | **0.3301** | 0.0463 | 0.1 |
| two_tower | 0.8000 | 0.2846 | 0.2911 | **0.1503** | 21.9 |

## music

1868 usuários, 4442 itens, 76843 positivos; 373 usuários avaliados.

| Modelo | HR@10 | Recall@20 | NDCG@10 | Cobertura@10 | Treino (s) |
|---|---:|---:|---:|---:|---:|
| popularity | 0.5576 | 0.1230 | 0.1593 | 0.0041 | 0.0 |
| itemknn | 0.9357 | 0.2890 | **0.4134** | 0.0943 | 0.1 |
| two_tower | **0.9437** | **0.2924** | 0.3986 | **0.2744** | 32.4 |
