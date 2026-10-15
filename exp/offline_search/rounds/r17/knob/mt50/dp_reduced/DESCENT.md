# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DP | 250 | 50/50 | 48.8% | 61.1% | 47.9% | 59.4% | 0.080 | False |
| DP | 500 | 50/50 | 49.9% | 61.8% | 47.9% | 59.4% | 0.073 | False |
| DP | 1000 | 50/50 | 50.1% | 61.9% | 47.9% | 59.4% | 0.056 | False |
| DP | 1500 | 50/50 | 51.6% | 63.0% | 47.9% | 59.4% | 0.054 | False |
| DP | 2000 | 50/50 | 52.5% | 63.8% | 47.9% | 59.4% | 0.053 | False |
| DP | 2500 | 50/50 | 52.5% | 63.5% | 47.9% | 59.4% | 0.051 | False |

## DP · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 21 | 21 | True |
| basketball-v3 | pure | k50 | 10 | 10 | True |
| bin-picking-v3 | k50 | k40 | 14 | 15 | True |
| box-close-v3 | pure | k50 | 9 | 9 | True |
| button-press-topdown-v3 | k30 | k20 | 48 | 50 | True |
| button-press-topdown-wall-v3 | k40 | k30 | 50 | 49 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | pure | k50 | 50 | 50 | True |
| coffee-button-v3 | k20 | k10 | 45 | 47 | True |
| coffee-pull-v3 | koff | none | 21 | 8 | True |
| coffee-push-v3 | k30 | k20 | 29 | 30 | True |
| dial-turn-v3 | koff | none | 34 | 18 | True |
| disassemble-v3 | pure | k50 | 33 | 33 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | k50 | k40 | 49 | 50 | True |
| door-unlock-v3 | pure | k50 | 49 | 49 | True |
| drawer-close-v3 | k20 | k10 | 50 | 50 | True |
| drawer-open-v3 | koff | none | 38 | 38 | True |
| faucet-close-v3 | k20 | k10 | 45 | 30 | True |
| faucet-open-v3 | k20 | k10 | 47 | 42 | True |
| hammer-v3 | k40 | k30 | 33 | 32 | True |
| hand-insert-v3 | koff | none | 11 | 12 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 48 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | pure | k50 | 4 | 4 | True |
| lever-pull-v3 | k30 | k20 | 20 | 21 | True |
| peg-insert-side-v3 | pure | k50 | 21 | 21 | True |
| peg-unplug-side-v3 | pure | k50 | 38 | 38 | True |
| pick-out-of-hole-v3 | pure | k50 | 17 | 17 | True |
| pick-place-v3 | pure | k50 | 30 | 30 | True |
| pick-place-wall-v3 | pure | k50 | 32 | 32 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 49 | True |
| plate-slide-back-v3 | koff | none | 46 | 42 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k30 | k20 | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | pure | k50 | 35 | 35 | True |
| push-wall-v3 | pure | k50 | 33 | 33 | True |
| reach-v3 | koff | none | 11 | 11 | True |
| reach-wall-v3 | k30 | k20 | 16 | 12 | True |
| shelf-place-v3 | k50 | k40 | 8 | 9 | True |
| soccer-v3 | pure | k50 | 9 | 9 | True |
| stick-pull-v3 | koff | none | 0 | 1 | True |
| stick-push-v3 | pure | k50 | 28 | 28 | True |
| sweep-into-v3 | koff | none | 7 | 7 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 48 | 50 | True |
## DP · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 21 | 21 | True |
| basketball-v3 | k50 | k40 | 8 | 10 | True |
| bin-picking-v3 | pure | k50 | 15 | 15 | True |
| box-close-v3 | k50 | k40 | 8 | 9 | True |
| button-press-topdown-v3 | k30 | k20 | 48 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 49 | 49 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | pure | k50 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 45 | 47 | True |
| coffee-pull-v3 | koff | none | 28 | 8 | True |
| coffee-push-v3 | k30 | k20 | 35 | 30 | True |
| dial-turn-v3 | koff | none | 28 | 18 | True |
| disassemble-v3 | pure | k50 | 33 | 33 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | k50 | k40 | 50 | 50 | True |
| door-unlock-v3 | pure | k50 | 49 | 49 | True |
| drawer-close-v3 | k20 | k10 | 50 | 50 | True |
| drawer-open-v3 | k20 | k10 | 42 | 38 | True |
| faucet-close-v3 | k20 | k10 | 35 | 30 | True |
| faucet-open-v3 | koff | none | 41 | 42 | True |
| hammer-v3 | k40 | k30 | 30 | 32 | True |
| hand-insert-v3 | koff | none | 15 | 12 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 48 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | pure | k50 | 4 | 4 | True |
| lever-pull-v3 | k30 | k20 | 19 | 21 | True |
| peg-insert-side-v3 | pure | k50 | 21 | 21 | True |
| peg-unplug-side-v3 | pure | k50 | 38 | 38 | True |
| pick-out-of-hole-v3 | pure | k50 | 17 | 17 | True |
| pick-place-v3 | pure | k50 | 30 | 30 | True |
| pick-place-wall-v3 | pure | k50 | 32 | 32 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 49 | True |
| plate-slide-back-v3 | koff | none | 48 | 42 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k30 | k20 | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | pure | k50 | 35 | 35 | True |
| push-wall-v3 | k30 | k20 | 32 | 33 | True |
| reach-v3 | koff | none | 12 | 11 | True |
| reach-wall-v3 | koff | none | 24 | 12 | True |
| shelf-place-v3 | k20 | k10 | 10 | 9 | True |
| soccer-v3 | pure | k50 | 9 | 9 | True |
| stick-pull-v3 | koff | none | 1 | 1 | True |
| stick-push-v3 | koff | none | 26 | 28 | True |
| sweep-into-v3 | koff | none | 19 | 7 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 49 | 50 | True |
## DP · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 20 | 21 | True |
| basketball-v3 | koff | none | 13 | 10 | True |
| bin-picking-v3 | k50 | k40 | 18 | 15 | True |
| box-close-v3 | pure | k50 | 9 | 9 | True |
| button-press-topdown-v3 | k30 | k20 | 49 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 49 | 49 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 48 | 47 | True |
| coffee-pull-v3 | koff | none | 23 | 8 | True |
| coffee-push-v3 | koff | none | 30 | 30 | True |
| dial-turn-v3 | koff | none | 32 | 18 | True |
| disassemble-v3 | pure | k50 | 33 | 33 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k30 | k20 | 20 | 21 | True |
| door-open-v3 | k50 | k40 | 48 | 50 | True |
| door-unlock-v3 | k30 | k20 | 48 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 36 | 38 | True |
| faucet-close-v3 | k10 | koff | 33 | 30 | True |
| faucet-open-v3 | koff | none | 49 | 42 | True |
| hammer-v3 | k40 | k30 | 30 | 32 | True |
| hand-insert-v3 | koff | none | 14 | 12 | True |
| handle-press-side-v3 | koff | none | 49 | 50 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | k50 | k40 | 2 | 4 | True |
| lever-pull-v3 | k20 | k10 | 21 | 21 | True |
| peg-insert-side-v3 | k40 | k30 | 20 | 21 | True |
| peg-unplug-side-v3 | pure | k50 | 38 | 38 | True |
| pick-out-of-hole-v3 | k50 | k40 | 16 | 17 | True |
| pick-place-v3 | pure | k50 | 30 | 30 | True |
| pick-place-wall-v3 | pure | k50 | 32 | 32 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 49 | True |
| plate-slide-back-v3 | koff | none | 47 | 42 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 35 | 35 | True |
| push-wall-v3 | k30 | k20 | 31 | 33 | True |
| reach-v3 | koff | none | 13 | 11 | True |
| reach-wall-v3 | koff | none | 20 | 12 | True |
| shelf-place-v3 | k50 | k40 | 7 | 9 | True |
| soccer-v3 | koff | none | 10 | 9 | True |
| stick-pull-v3 | koff | none | 5 | 1 | True |
| stick-push-v3 | pure | k50 | 28 | 28 | True |
| sweep-into-v3 | koff | none | 25 | 7 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 49 | 50 | True |
## DP · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 21 | 21 | True |
| basketball-v3 | koff | none | 11 | 10 | True |
| bin-picking-v3 | koff | none | 22 | 15 | True |
| box-close-v3 | pure | k50 | 9 | 9 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k50 | k40 | 48 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 47 | True |
| coffee-pull-v3 | koff | none | 26 | 8 | True |
| coffee-push-v3 | koff | none | 38 | 30 | True |
| dial-turn-v3 | koff | none | 21 | 18 | True |
| disassemble-v3 | pure | k50 | 33 | 33 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k50 | k40 | 21 | 21 | True |
| door-open-v3 | k30 | k20 | 49 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 45 | 38 | True |
| faucet-close-v3 | koff | none | 40 | 30 | True |
| faucet-open-v3 | koff | none | 48 | 42 | True |
| hammer-v3 | k40 | k30 | 31 | 32 | True |
| hand-insert-v3 | koff | none | 17 | 12 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | pure | k50 | 4 | 4 | True |
| lever-pull-v3 | k50 | k40 | 21 | 21 | True |
| peg-insert-side-v3 | k30 | k20 | 23 | 21 | True |
| peg-unplug-side-v3 | pure | k50 | 38 | 38 | True |
| pick-out-of-hole-v3 | pure | k50 | 17 | 17 | True |
| pick-place-v3 | pure | k50 | 30 | 30 | True |
| pick-place-wall-v3 | k30 | k20 | 31 | 32 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 49 | True |
| plate-slide-back-v3 | koff | none | 46 | 42 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k20 | k10 | 35 | 35 | True |
| push-wall-v3 | k20 | k10 | 31 | 33 | True |
| reach-v3 | koff | none | 12 | 11 | True |
| reach-wall-v3 | koff | none | 17 | 12 | True |
| shelf-place-v3 | koff | none | 8 | 9 | True |
| soccer-v3 | k30 | k20 | 12 | 9 | True |
| stick-pull-v3 | koff | none | 9 | 1 | True |
| stick-push-v3 | pure | k50 | 28 | 28 | True |
| sweep-into-v3 | koff | none | 18 | 7 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 50 | 50 | True |
## DP · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k40 | k30 | 19 | 21 | True |
| basketball-v3 | koff | none | 12 | 10 | True |
| bin-picking-v3 | koff | none | 22 | 15 | True |
| box-close-v3 | pure | k50 | 9 | 9 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | pure | k50 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 47 | True |
| coffee-pull-v3 | koff | none | 27 | 8 | True |
| coffee-push-v3 | koff | none | 41 | 30 | True |
| dial-turn-v3 | koff | none | 22 | 18 | True |
| disassemble-v3 | pure | k50 | 33 | 33 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k50 | k40 | 24 | 21 | True |
| door-open-v3 | k20 | k10 | 49 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 45 | 38 | True |
| faucet-close-v3 | k20 | k10 | 40 | 30 | True |
| faucet-open-v3 | koff | none | 48 | 42 | True |
| hammer-v3 | k40 | k30 | 30 | 32 | True |
| hand-insert-v3 | koff | none | 20 | 12 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | pure | k50 | 4 | 4 | True |
| lever-pull-v3 | koff | none | 19 | 21 | True |
| peg-insert-side-v3 | k30 | k20 | 24 | 21 | True |
| peg-unplug-side-v3 | pure | k50 | 38 | 38 | True |
| pick-out-of-hole-v3 | pure | k50 | 17 | 17 | True |
| pick-place-v3 | pure | k50 | 30 | 30 | True |
| pick-place-wall-v3 | pure | k50 | 32 | 32 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 49 | True |
| plate-slide-back-v3 | koff | none | 50 | 42 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 50 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k10 | koff | 33 | 35 | True |
| push-wall-v3 | k10 | koff | 34 | 33 | True |
| reach-v3 | koff | none | 12 | 11 | True |
| reach-wall-v3 | k20 | k10 | 15 | 12 | True |
| shelf-place-v3 | koff | none | 7 | 9 | True |
| soccer-v3 | k20 | k10 | 7 | 9 | True |
| stick-pull-v3 | koff | none | 8 | 1 | True |
| stick-push-v3 | k30 | k20 | 34 | 28 | True |
| sweep-into-v3 | koff | none | 21 | 7 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 50 | 50 | True |
## DP · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k40 | k30 | 22 | 21 | True |
| basketball-v3 | koff | none | 14 | 10 | True |
| bin-picking-v3 | koff | none | 23 | 15 | True |
| box-close-v3 | k40 | k30 | 8 | 9 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | pure | k50 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 47 | True |
| coffee-pull-v3 | koff | none | 26 | 8 | True |
| coffee-push-v3 | koff | none | 39 | 30 | True |
| dial-turn-v3 | k40 | k30 | 19 | 18 | True |
| disassemble-v3 | pure | k50 | 33 | 33 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k40 | k30 | 19 | 21 | True |
| door-open-v3 | k40 | k30 | 48 | 50 | True |
| door-unlock-v3 | k10 | koff | 47 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 45 | 38 | True |
| faucet-close-v3 | koff | none | 44 | 30 | True |
| faucet-open-v3 | koff | none | 43 | 42 | True |
| hammer-v3 | k40 | k30 | 31 | 32 | True |
| hand-insert-v3 | koff | none | 24 | 12 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | pure | k50 | 4 | 4 | True |
| lever-pull-v3 | koff | none | 26 | 21 | True |
| peg-insert-side-v3 | k30 | k20 | 25 | 21 | True |
| peg-unplug-side-v3 | pure | k50 | 38 | 38 | True |
| pick-out-of-hole-v3 | pure | k50 | 17 | 17 | True |
| pick-place-v3 | pure | k50 | 30 | 30 | True |
| pick-place-wall-v3 | k40 | k30 | 30 | 32 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 49 | True |
| plate-slide-back-v3 | koff | none | 50 | 42 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k20 | k10 | 36 | 35 | True |
| push-wall-v3 | k20 | k10 | 32 | 33 | True |
| reach-v3 | koff | none | 13 | 11 | True |
| reach-wall-v3 | koff | none | 10 | 12 | True |
| shelf-place-v3 | koff | none | 12 | 9 | True |
| soccer-v3 | k30 | k20 | 9 | 9 | True |
| stick-pull-v3 | koff | none | 8 | 1 | True |
| stick-push-v3 | k10 | koff | 26 | 28 | True |
| sweep-into-v3 | koff | none | 17 | 7 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 50 | 50 | True |
