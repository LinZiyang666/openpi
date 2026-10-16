# Frozen-fit failed-episode removal

Primary includes empty tasks as failures. Secondary uses the same non-empty tasks in both arms. MT50 reports equal-weight difficulty-group and task means. Missing simulated outcomes remain pending.

| Model / suite / size | Kept | Primary filtered / unfiltered task SR | Historical unfiltered SR | MT50 group SR filtered / unfiltered | Secondary filtered / unfiltered task SR | Empty tasks |
|---|---:|---|---|---|---|---|
| pi05/mt50/250 | 137 | 34.16% / 34.84% | 34.84% | 19.69% / 21.55% | 40.67% / 40.48% | hammer-v3, hand-insert-v3, handle-pull-side-v3, handle-pull-v3, reach-v3, soccer-v3, stick-pull-v3, sweep-v3 |
| pi05/mt50/500 | 282 | 39.44% / 41.60% | 41.60% | 25.55% / 30.02% | 42.87% / 45.22% | handle-pull-v3, soccer-v3, stick-pull-v3, sweep-v3 |
| pi05/mt50/1000 | 562 | 45.24% / 47.44% | 47.44% | 32.57% / 37.76% | 48.13% / 50.47% | handle-pull-v3, soccer-v3, stick-pull-v3 |
| pi05/mt50/1500 | 849 | 50.04% / 50.12% | 50.12% | 38.69% / 39.48% | 52.12% / 52.21% | handle-pull-v3, stick-pull-v3 |
| pi05/mt50/2000 | 1122 | 50.72% / 50.56% | 50.56% | 39.78% / 40.20% | 52.83% / 52.67% | handle-pull-v3, stick-pull-v3 |
| pi05/mt50/2500 | 1425 | 52.32% / 50.44% | 50.44% | 41.32% / 40.02% | 54.50% / 52.54% | handle-pull-v3, stick-pull-v3 |
| groot/mt50/250 | 186 | 36.52% / 36.12% | 36.12% | 24.35% / 23.04% | 38.85% / 38.38% | coffee-pull-v3, push-back-v3, stick-pull-v3 |
| groot/mt50/500 | 379 | 45.80% / 46.08% | 46.08% | 33.16% / 33.71% | 47.71% / 47.92% | push-back-v3, stick-pull-v3 |
| groot/mt50/1000 | 762 | 52.60% / 52.40% | 52.40% | 38.83% / 38.52% | 53.67% / 53.47% | stick-pull-v3 |
| groot/mt50/1500 | 1140 | 56.60% / 56.16% | 56.16% | 42.92% / 42.08% | 57.76% / 57.31% | stick-pull-v3 |
| groot/mt50/2000 | 1516 | 57.20% / 57.44% | 57.44% | 43.93% / 43.88% | 58.37% / 58.61% | stick-pull-v3 |
| groot/mt50/2500 | 1899 | 58.80% / 59.32% | 59.32% | 46.20% / 46.55% | 60.00% / 60.53% | stick-pull-v3 |
| dp/mt50/250 | 153 | 39.56% / 43.36% | 43.36% | 22.68% / 27.87% | 44.95% / 48.68% | assembly-v3, handle-pull-v3, pick-out-of-hole-v3, push-back-v3, soccer-v3, stick-pull-v3 |
| dp/mt50/500 | 312 | 45.20% / 49.36% | 49.36% | 29.46% / 35.89% | 49.13% / 53.22% | assembly-v3, push-back-v3, soccer-v3, stick-pull-v3 |
| dp/mt50/1000 | 623 | 51.68% / 54.92% | 54.92% | 34.90% / 40.38% | 54.98% / 58.13% | assembly-v3, push-back-v3, stick-pull-v3 |
| dp/mt50/1500 | 943 | 56.64% / 57.24% | 57.24% | 40.86% / 41.60% | 60.26% / 60.72% | assembly-v3, push-back-v3, stick-pull-v3 |
| dp/mt50/2000 | 1262 | 58.32% / 57.44% | 57.44% | 43.18% / 42.19% | 59.51% / 58.61% | push-back-v3 |
| dp/mt50/2500 | 1577 | 59.68% / 57.80% | 57.80% | 45.76% / 43.02% | 60.90% / 58.98% | push-back-v3 |
| fastwam/mt50/250 | 144 | 34.04% / 35.76% | 35.76% | 19.02% / 21.46% | 41.51% / 43.02% | assembly-v3, basketball-v3, box-close-v3, dial-turn-v3, handle-pull-side-v3, peg-insert-side-v3, push-back-v3, shelf-place-v3, stick-pull-v3 |
| fastwam/mt50/500 | 295 | 38.16% / 39.80% | 39.80% | 22.54% / 25.32% | 40.60% / 42.34% | dial-turn-v3, push-back-v3, stick-pull-v3 |
| fastwam/mt50/1000 | 590 | 44.08% / 44.72% | 44.72% | 27.32% / 30.00% | 46.89% / 47.45% | dial-turn-v3, push-back-v3, stick-pull-v3 |
| fastwam/mt50/1500 | 874 | 47.32% / 46.24% | 46.24% | 30.54% / 31.30% | 50.34% / 49.15% | dial-turn-v3, push-back-v3, stick-pull-v3 |
| fastwam/mt50/2000 | 1159 | 47.32% / 47.24% | 47.24% | 30.65% / 31.87% | 49.29% / 49.21% | push-back-v3, stick-pull-v3 |
| fastwam/mt50/2500 | 1457 | 49.36% / 48.84% | 48.84% | 32.24% / 33.80% | 51.42% / 50.88% | push-back-v3, stick-pull-v3 |
| dp/libero_spatial/50 | 42 | 69.60% / 70.20% | 72.20% | — | 69.60% / 70.20% |  |
| dp/libero_spatial/100 | 84 | 88.40% / 85.80% | 86.60% | — | 88.40% / 85.80% |  |
| dp/libero_spatial/200 | 177 | 92.00% / 87.60% | 88.40% | — | 92.00% / 87.60% |  |
| dp/libero_spatial/300 | 265 | 94.00% / 92.40% | 91.80% | — | 94.00% / 92.40% |  |
| dp/libero_spatial/400 | 349 | 94.80% / 92.80% | 93.00% | — | 94.80% / 92.80% |  |
| dp/libero_spatial/500 | 436 | 93.80% / 94.60% | 94.60% | — | 93.80% / 94.60% |  |
| dp/libero_10/50 | 42 | 32.00% / 23.40% | 24.20% | — | 32.00% / 23.40% |  |
| dp/libero_10/100 | 82 | 57.60% / 45.20% | 46.60% | — | 57.60% / 45.20% |  |
| dp/libero_10/200 | 164 | 66.60% / 48.00% | 48.80% | — | 66.60% / 48.00% |  |
| dp/libero_10/300 | 245 | 70.80% / 59.40% | 57.80% | — | 70.80% / 59.40% |  |
| dp/libero_10/400 | 330 | 73.60% / 61.80% | 61.20% | — | 73.60% / 61.80% |  |
| dp/libero_10/500 | 405 | 74.40% / 64.80% | 64.80% | — | 74.40% / 64.80% |  |
| fastwam/libero_spatial/50 | 43 | 68.80% / 78.60% | 78.60% | — | 68.80% / 78.60% |  |
| fastwam/libero_spatial/100 | 85 | 80.40% / 83.20% | 83.20% | — | 80.40% / 83.20% |  |
| fastwam/libero_spatial/200 | 175 | 88.80% / 84.40% | 84.40% | — | 88.80% / 84.40% |  |
| fastwam/libero_spatial/300 | 259 | 88.60% / 84.80% | 84.80% | — | 88.60% / 84.80% |  |
| fastwam/libero_spatial/400 | 345 | pending / 85.40% | 85.40% | — | pending / 85.40% |  |
| fastwam/libero_spatial/500 | 432 | 87.80% / 87.80% | 87.80% | — | 87.80% / 87.80% |  |
| fastwam/libero_10/50 | 49 | 39.20% / 39.00% | 39.00% | — | 39.20% / 39.00% |  |
| fastwam/libero_10/100 | 95 | 46.20% / 51.00% | 51.00% | — | 46.20% / 51.00% |  |
| fastwam/libero_10/200 | 188 | 55.60% / 56.60% | 56.60% | — | 55.60% / 56.60% |  |
| fastwam/libero_10/300 | 279 | pending / 62.60% | 62.60% | — | pending / 62.60% |  |
| fastwam/libero_10/400 | 371 | pending / 64.80% | 64.80% | — | pending / 64.80% |  |
| fastwam/libero_10/500 | 462 | pending / 69.00% | 69.00% | — | pending / 69.00% |  |
| cosmos/libero_spatial/50 | 45 | pending / 11.20% | 11.20% | — | pending / 11.20% |  |
| cosmos/libero_spatial/100 | 92 | pending / 85.60% | 85.60% | — | pending / 85.60% |  |
| cosmos/libero_spatial/200 | 189 | pending / 93.60% | 93.60% | — | pending / 93.60% |  |
| cosmos/libero_spatial/300 | 281 | pending / 95.20% | 95.20% | — | pending / 95.20% |  |
| cosmos/libero_spatial/400 | 373 | pending / 94.00% | 94.00% | — | pending / 94.00% |  |
| cosmos/libero_spatial/500 | 465 | 96.00% / 95.00% | 95.00% | — | 96.00% / 95.00% |  |
| cosmos/libero_10/50 | 49 | pending / 12.00% | 12.00% | — | pending / 12.00% |  |
| cosmos/libero_10/100 | 98 | pending / 45.00% | 45.00% | — | pending / 45.00% |  |
| cosmos/libero_10/200 | 194 | pending / 66.00% | 66.00% | — | pending / 66.00% |  |
| cosmos/libero_10/300 | 287 | pending / 70.60% | 70.60% | — | pending / 70.60% |  |
| cosmos/libero_10/400 | 383 | pending / 77.20% | 77.20% | — | pending / 77.20% |  |
| cosmos/libero_10/500 | 479 | 74.00% / 75.20% | 75.20% | — | 74.00% / 75.20% |  |
| pi05/libero_spatial/50 | 46 | 85.80% / 81.80% | 81.80% | — | 85.80% / 81.80% |  |
| pi05/libero_spatial/100 | 94 | 94.60% / 95.40% | 95.40% | — | 94.60% / 95.40% |  |
| pi05/libero_spatial/200 | 193 | 96.00% / 97.20% | 97.20% | — | 96.00% / 97.20% |  |
| pi05/libero_spatial/300 | 291 | 96.60% / 98.00% | 98.00% | — | 96.60% / 98.00% |  |
| pi05/libero_spatial/400 | 389 | 97.40% / 98.00% | 98.00% | — | 97.40% / 98.00% |  |
| pi05/libero_spatial/500 | 487 | 97.00% / 96.80% | 96.80% | — | 97.00% / 96.80% |  |
| pi05/libero_10/50 | 44 | 64.20% / 63.40% | 63.40% | — | 64.20% / 63.40% |  |
| pi05/libero_10/100 | 89 | 69.80% / 69.20% | 69.20% | — | 69.80% / 69.20% |  |
| pi05/libero_10/200 | 181 | 78.40% / 77.20% | 77.20% | — | 78.40% / 77.20% |  |
| pi05/libero_10/300 | 266 | 79.20% / 79.40% | 79.40% | — | 79.20% / 79.40% |  |
| pi05/libero_10/400 | 350 | 84.20% / 84.40% | 84.40% | — | 84.20% / 84.40% |  |
| pi05/libero_10/500 | 436 | 83.20% / 83.60% | 83.60% | — | 83.20% / 83.60% |  |
| groot/libero_spatial/50 | 45 | 76.80% / 79.40% | 79.40% | — | 76.80% / 79.40% |  |
| groot/libero_spatial/100 | 91 | 95.20% / 95.60% | 95.60% | — | 95.20% / 95.60% |  |
| groot/libero_spatial/200 | 186 | 94.00% / 94.60% | 94.60% | — | 94.00% / 94.60% |  |
| groot/libero_spatial/300 | 277 | 97.00% / 97.00% | 97.00% | — | 97.00% / 97.00% |  |
| groot/libero_spatial/400 | 367 | 95.80% / 95.60% | 95.60% | — | 95.80% / 95.60% |  |
| groot/libero_spatial/500 | 456 | 96.00% / 96.40% | 96.40% | — | 96.00% / 96.40% |  |
| groot/libero_10/50 | 44 | 59.60% / 60.80% | 60.80% | — | 59.60% / 60.80% |  |
| groot/libero_10/100 | 90 | 67.20% / 64.80% | 64.80% | — | 67.20% / 64.80% |  |
| groot/libero_10/200 | 175 | 73.40% / 73.00% | 73.00% | — | 73.40% / 73.00% |  |
| groot/libero_10/300 | 261 | 75.80% / 76.00% | 76.00% | — | 75.80% / 76.00% |  |
| groot/libero_10/400 | 346 | 79.60% / 78.60% | 78.60% | — | 79.60% / 78.60% |  |
| groot/libero_10/500 | 427 | 82.80% / 81.00% | 81.00% | — | 82.80% / 81.00% |  |
