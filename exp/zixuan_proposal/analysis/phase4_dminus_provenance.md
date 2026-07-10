# TRACER Phase 4 — D- provenance manifest (held-out init split)

Recovery formula: `task_id = episode_id // 50`, `init_state_idx = episode_id % 50` (num_trials_per_task=50, seed 7). **init_state_idx indexes into the HELD-OUT pool** `exp/common/data/db_init/libero/<suite>/<task>.init` (50 states/task = full `.init` minus `.pruned_init` eval set), NOT the pruned eval set. This is the contamination-fixed run (the earlier D- run used pruned_init = the eval set and is superseded).

## libero_spatial — 18 failure episodes (792 D- steps, 44 replan-cycles each = full max_steps timeout)

Failures per task_id: {0: 8, 4: 3, 5: 1, 8: 5, 9: 1}

| episode_id | task_id | init_state_idx |
|---|---|---|
| 5 | 0 | 5 |
| 7 | 0 | 7 |
| 9 | 0 | 9 |
| 10 | 0 | 10 |
| 17 | 0 | 17 |
| 20 | 0 | 20 |
| 38 | 0 | 38 |
| 40 | 0 | 40 |
| 206 | 4 | 6 |
| 219 | 4 | 19 |
| 247 | 4 | 47 |
| 275 | 5 | 25 |
| 400 | 8 | 0 |
| 402 | 8 | 2 |
| 403 | 8 | 3 |
| 406 | 8 | 6 |
| 410 | 8 | 10 |
| 475 | 9 | 25 |

## libero_10 — 85 failure episodes (8840 D- steps, 104 replan-cycles each = full max_steps timeout)

Failures per task_id: {0: 3, 2: 5, 3: 8, 4: 4, 5: 10, 6: 9, 7: 2, 8: 28, 9: 16}

| episode_id | task_id | init_state_idx |
|---|---|---|
| 8 | 0 | 8 |
| 26 | 0 | 26 |
| 27 | 0 | 27 |
| 105 | 2 | 5 |
| 107 | 2 | 7 |
| 128 | 2 | 28 |
| 137 | 2 | 37 |
| 139 | 2 | 39 |
| 160 | 3 | 10 |
| 173 | 3 | 23 |
| 175 | 3 | 25 |
| 179 | 3 | 29 |
| 183 | 3 | 33 |
| 186 | 3 | 36 |
| 188 | 3 | 38 |
| 193 | 3 | 43 |
| 226 | 4 | 26 |
| 233 | 4 | 33 |
| 236 | 4 | 36 |
| 238 | 4 | 38 |
| 259 | 5 | 9 |
| 261 | 5 | 11 |
| 264 | 5 | 14 |
| 267 | 5 | 17 |
| 277 | 5 | 27 |
| 281 | 5 | 31 |
| 287 | 5 | 37 |
| 289 | 5 | 39 |
| 290 | 5 | 40 |
| 294 | 5 | 44 |
| 302 | 6 | 2 |
| 304 | 6 | 4 |
| 307 | 6 | 7 |
| 308 | 6 | 8 |
| 315 | 6 | 15 |
| 316 | 6 | 16 |
| 320 | 6 | 20 |
| 321 | 6 | 21 |
| 336 | 6 | 36 |
| 356 | 7 | 6 |
| 382 | 7 | 32 |
| 403 | 8 | 3 |
| 404 | 8 | 4 |
| 406 | 8 | 6 |
| 407 | 8 | 7 |
| 408 | 8 | 8 |
| 412 | 8 | 12 |
| 413 | 8 | 13 |
| 414 | 8 | 14 |
| 415 | 8 | 15 |
| 416 | 8 | 16 |
| 417 | 8 | 17 |
| 421 | 8 | 21 |
| 423 | 8 | 23 |
| 424 | 8 | 24 |
| 425 | 8 | 25 |
| 428 | 8 | 28 |
| 429 | 8 | 29 |
| 430 | 8 | 30 |
| 431 | 8 | 31 |
| 432 | 8 | 32 |
| 433 | 8 | 33 |
| 435 | 8 | 35 |
| 438 | 8 | 38 |
| 439 | 8 | 39 |
| 440 | 8 | 40 |
| 442 | 8 | 42 |
| 448 | 8 | 48 |
| 449 | 8 | 49 |
| 454 | 9 | 4 |
| 455 | 9 | 5 |
| 461 | 9 | 11 |
| 467 | 9 | 17 |
| 469 | 9 | 19 |
| 470 | 9 | 20 |
| 474 | 9 | 24 |
| 477 | 9 | 27 |
| 478 | 9 | 28 |
| 482 | 9 | 32 |
| 484 | 9 | 34 |
| 486 | 9 | 36 |
| 487 | 9 | 37 |
| 488 | 9 | 38 |
| 490 | 9 | 40 |
| 499 | 9 | 49 |
