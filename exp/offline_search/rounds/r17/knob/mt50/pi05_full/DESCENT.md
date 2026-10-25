# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| π0.5 | 250 | 50/50 | 46.9% | 56.3% | 46.7% | 56.3% | 0.671 | False |
| π0.5 | 500 | 50/50 | 46.8% | 56.9% | 46.7% | 56.3% | 0.601 | False |
| π0.5 | 1000 | 50/50 | 49.1% | 58.1% | 46.7% | 56.3% | 0.552 | False |
| π0.5 | 1500 | 50/50 | 49.4% | 58.9% | 46.7% | 56.3% | 0.515 | False |
| π0.5 | 2000 | 50/50 | 49.1% | 58.4% | 46.7% | 56.3% | 0.495 | False |
| π0.5 | 2500 | 50/50 | 49.3% | 58.6% | 46.7% | 56.3% | 0.483 | False |

## π0.5 · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | pure | k50 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 48 | 49 | True |
| button-press-v3 | k50 | k40 | 48 | 50 | True |
| button-press-wall-v3 | k10 | koff | 49 | 50 | True |
| coffee-button-v3 | pure | k50 | 49 | 49 | True |
| coffee-pull-v3 | pure | k50 | 40 | 40 | True |
| coffee-push-v3 | pure | k50 | 30 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 24 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | pure | k50 | 49 | 49 | True |
| door-unlock-v3 | pure | k50 | 38 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | koff | none | 45 | 43 | True |
| faucet-close-v3 | k50 | k40 | 31 | 26 | True |
| faucet-open-v3 | k20 | k10 | 48 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | pure | k50 | 11 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 48 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 16 | 16 | True |
| peg-insert-side-v3 | koff | none | 1 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | pure | k50 | 8 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 40 | 33 | True |
| plate-slide-back-v3 | koff | none | 7 | 7 | True |
| plate-slide-side-v3 | koff | none | 45 | 47 | True |
| plate-slide-v3 | koff | none | 30 | 31 | True |
| push-back-v3 | pure | k50 | 30 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | k10 | koff | 16 | 16 | True |
| reach-wall-v3 | pure | k50 | 45 | 45 | True |
| shelf-place-v3 | pure | k50 | 16 | 16 | True |
| soccer-v3 | pure | k50 | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 41 | 41 | True |
| sweep-into-v3 | pure | k50 | 21 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 48 | 50 | True |
| window-open-v3 | k50 | k40 | 43 | 45 | True |
## π0.5 · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | pure | k50 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | k50 | k40 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 48 | 50 | True |
| button-press-topdown-wall-v3 | k10 | koff | 47 | 49 | True |
| button-press-v3 | k50 | k40 | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | k50 | k40 | 47 | 49 | True |
| coffee-pull-v3 | k50 | k40 | 39 | 40 | True |
| coffee-push-v3 | k30 | k20 | 29 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | k10 | koff | 20 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | pure | k50 | 49 | 49 | True |
| door-unlock-v3 | pure | k50 | 38 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k20 | k10 | 45 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | k30 | k20 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k20 | k10 | 10 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | k30 | k20 | 21 | 16 | True |
| peg-insert-side-v3 | koff | none | 4 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | k40 | k30 | 6 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | pure | k50 | 7 | 7 | True |
| plate-slide-side-v3 | koff | none | 47 | 47 | True |
| plate-slide-v3 | koff | none | 30 | 31 | True |
| push-back-v3 | k40 | k30 | 30 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | k50 | k40 | 15 | 16 | True |
| reach-wall-v3 | k40 | k30 | 45 | 45 | True |
| shelf-place-v3 | pure | k50 | 16 | 16 | True |
| soccer-v3 | pure | k50 | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 41 | 41 | True |
| sweep-into-v3 | koff | none | 28 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k50 | k40 | 44 | 45 | True |
## π0.5 · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | k50 | k40 | 13 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 49 | True |
| button-press-v3 | k20 | k10 | 48 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 48 | 50 | True |
| coffee-button-v3 | k40 | k30 | 49 | 49 | True |
| coffee-pull-v3 | k40 | k30 | 40 | 40 | True |
| coffee-push-v3 | k50 | k40 | 29 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 31 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | koff | none | 49 | 49 | True |
| door-unlock-v3 | koff | none | 43 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k30 | k20 | 43 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k40 | k30 | 12 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 16 | 16 | True |
| peg-insert-side-v3 | koff | none | 1 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | k30 | k20 | 10 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | pure | k50 | 7 | 7 | True |
| plate-slide-side-v3 | koff | none | 49 | 47 | True |
| plate-slide-v3 | koff | none | 33 | 31 | True |
| push-back-v3 | koff | none | 29 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | koff | none | 17 | 16 | True |
| reach-wall-v3 | k50 | k40 | 43 | 45 | True |
| shelf-place-v3 | k50 | k40 | 14 | 16 | True |
| soccer-v3 | pure | k50 | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 47 | 41 | True |
| sweep-into-v3 | koff | none | 34 | 21 | True |
| sweep-v3 | k50 | k40 | 12 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k50 | k40 | 45 | 45 | True |
## π0.5 · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 20 | 19 | True |
| basketball-v3 | k40 | k30 | 12 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 49 | 50 | True |
| coffee-button-v3 | koff | none | 49 | 49 | True |
| coffee-pull-v3 | k40 | k30 | 41 | 40 | True |
| coffee-push-v3 | k50 | k40 | 32 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 28 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k40 | k30 | 13 | 14 | True |
| door-open-v3 | koff | none | 50 | 49 | True |
| door-unlock-v3 | koff | none | 49 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k30 | k20 | 43 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k40 | k30 | 11 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 18 | 16 | True |
| peg-insert-side-v3 | koff | none | 2 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | k40 | k30 | 8 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | pure | k50 | 7 | 7 | True |
| plate-slide-side-v3 | koff | none | 49 | 47 | True |
| plate-slide-v3 | koff | none | 35 | 31 | True |
| push-back-v3 | koff | none | 30 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | koff | none | 17 | 16 | True |
| reach-wall-v3 | k30 | k20 | 45 | 45 | True |
| shelf-place-v3 | pure | k50 | 16 | 16 | True |
| soccer-v3 | k50 | k40 | 2 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 41 | True |
| sweep-into-v3 | koff | none | 35 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 45 | 45 | True |
## π0.5 · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 19 | 19 | True |
| basketball-v3 | k50 | k40 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | koff | none | 47 | 49 | True |
| coffee-pull-v3 | k20 | k10 | 38 | 40 | True |
| coffee-push-v3 | k30 | k20 | 34 | 30 | True |
| dial-turn-v3 | koff | none | 1 | 1 | True |
| disassemble-v3 | koff | none | 28 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k40 | k30 | 12 | 14 | True |
| door-open-v3 | koff | none | 49 | 49 | True |
| door-unlock-v3 | koff | none | 48 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k10 | koff | 41 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k40 | k30 | 9 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 19 | 16 | True |
| peg-insert-side-v3 | koff | none | 2 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 6 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | pure | k50 | 7 | 7 | True |
| plate-slide-side-v3 | koff | none | 48 | 47 | True |
| plate-slide-v3 | koff | none | 34 | 31 | True |
| push-back-v3 | koff | none | 31 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | koff | none | 15 | 16 | True |
| reach-wall-v3 | k20 | k10 | 44 | 45 | True |
| shelf-place-v3 | pure | k50 | 16 | 16 | True |
| soccer-v3 | pure | k50 | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 41 | True |
| sweep-into-v3 | koff | none | 35 | 21 | True |
| sweep-v3 | k50 | k40 | 13 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k50 | k40 | 44 | 45 | True |
## π0.5 · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | koff | none | 18 | 19 | True |
| basketball-v3 | pure | k50 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 49 | 49 | True |
| coffee-pull-v3 | k40 | k30 | 38 | 40 | True |
| coffee-push-v3 | k50 | k40 | 35 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 30 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k30 | k20 | 12 | 14 | True |
| door-open-v3 | koff | none | 50 | 49 | True |
| door-unlock-v3 | koff | none | 49 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | koff | none | 42 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k40 | k30 | 11 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 18 | 16 | True |
| peg-insert-side-v3 | koff | none | 1 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | k30 | k20 | 7 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | koff | none | 5 | 7 | True |
| plate-slide-side-v3 | koff | none | 48 | 47 | True |
| plate-slide-v3 | koff | none | 35 | 31 | True |
| push-back-v3 | koff | none | 33 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | koff | none | 17 | 16 | True |
| reach-wall-v3 | koff | none | 43 | 45 | True |
| shelf-place-v3 | k50 | k40 | 14 | 16 | True |
| soccer-v3 | pure | k50 | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 41 | True |
| sweep-into-v3 | koff | none | 33 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k50 | k40 | 43 | 45 | True |
