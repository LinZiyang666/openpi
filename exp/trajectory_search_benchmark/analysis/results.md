# Trajectory Search Benchmark — plan §7.4 / §10 #4

## Matrix (single step, median over 10 repeats)

| entries | depth | fusion | legacy ms | new memo OFF ms | new memo ON ms | speedup (legacy → memo ON) |
|---|---|---|---|---|---|---|
| 5000 | 3 | weighted_rrf | 68.96 | 58.38 | 54.72 | 1.26x |
| 5000 | 3 | weighted_score_sum | 61.92 | 54.80 | 52.28 | 1.18x |
| 5000 | 5 | weighted_rrf | 105.29 | 91.40 | 80.00 | 1.32x |
| 5000 | 5 | weighted_score_sum | 95.55 | 94.17 | 78.54 | 1.22x |
| 20000 | 3 | weighted_rrf | 304.88 | 240.73 | 211.68 | 1.44x |
| 20000 | 3 | weighted_score_sum | 306.09 | 242.04 | 204.92 | 1.49x |
| 20000 | 5 | weighted_rrf | 493.72 | 368.73 | 315.68 | 1.56x |
| 20000 | 5 | weighted_score_sum | 474.54 | 364.61 | 321.09 | 1.48x |

## Score-memo hit curve (5 successive steps, depth=3, score_sum, 5_000 entries)

| step | legacy ms | new memo OFF ms | new memo ON ms |
|---|---|---|---|
| 0 | 56.16 | 63.01 | 54.46 |
| 1 | 62.72 | 58.95 | 52.97 |
| 2 | 55.55 | 63.74 | 50.28 |
| 3 | 60.66 | 56.53 | 52.46 |
| 4 | 62.57 | 53.52 | 44.62 |
