# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| π0.5 | 250 | 50/50 | 47.1% | 57.3% | 46.7% | 56.3% | 0.842 | False |
| π0.5 | 500 | 50/50 | 47.8% | 58.3% | 46.7% | 56.3% | 0.789 | False |
| π0.5 | 1000 | 50/50 | 49.5% | 58.6% | 46.7% | 56.3% | 0.725 | False |
| π0.5 | 1500 | 50/50 | 49.6% | 58.8% | 46.7% | 56.3% | 0.662 | False |
| π0.5 | 2000 | 50/50 | 50.3% | 59.6% | 46.7% | 56.3% | 0.677 | False |
| π0.5 | 2500 | 50/50 | 50.9% | 60.3% | 46.7% | 56.3% | 0.626 | False |

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
| disassemble-v3 | koff | none | 21 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | pure | k50 | 49 | 49 | True |
| door-unlock-v3 | pure | k50 | 38 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k10 | koff | 43 | 43 | True |
| faucet-close-v3 | k40 | k30 | 24 | 26 | True |
| faucet-open-v3 | k20 | k10 | 48 | 50 | True |
| hammer-v3 | koff | none | 9 | 9 | True |
| hand-insert-v3 | koff | none | 11 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 48 | 50 | True |
| handle-pull-side-v3 | koff | none | 1 | 0 | True |
| handle-pull-v3 | koff | none | 1 | 0 | True |
| lever-pull-v3 | pure | k50 | 16 | 16 | True |
| peg-insert-side-v3 | koff | none | 0 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | pure | k50 | 8 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 41 | 33 | True |
| plate-slide-back-v3 | koff | none | 40 | 7 | True |
| plate-slide-side-v3 | koff | none | 45 | 47 | True |
| plate-slide-v3 | koff | none | 29 | 31 | True |
| push-back-v3 | pure | k50 | 30 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | koff | none | 15 | 16 | True |
| reach-wall-v3 | pure | k50 | 45 | 45 | True |
| shelf-place-v3 | pure | k50 | 16 | 16 | True |
| soccer-v3 | koff | none | 5 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 41 | 41 | True |
| sweep-into-v3 | pure | k50 | 21 | 21 | True |
| sweep-v3 | koff | none | 13 | 11 | True |
| window-close-v3 | k10 | koff | 48 | 50 | True |
| window-open-v3 | k50 | k40 | 43 | 45 | True |
## π0.5 · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | pure | k50 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | k10 | koff | 50 | 50 | True |
| button-press-topdown-wall-v3 | k10 | koff | 48 | 49 | True |
| button-press-v3 | k50 | k40 | 49 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | pure | k50 | 49 | 49 | True |
| coffee-pull-v3 | k50 | k40 | 40 | 40 | True |
| coffee-push-v3 | k50 | k40 | 28 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 22 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | koff | none | 48 | 49 | True |
| door-unlock-v3 | pure | k50 | 38 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k30 | k20 | 41 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | k30 | k20 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | pure | k50 | 11 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 16 | 16 | True |
| peg-insert-side-v3 | koff | none | 0 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | pure | k50 | 8 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | koff | none | 43 | 7 | True |
| plate-slide-side-v3 | koff | none | 47 | 47 | True |
| plate-slide-v3 | koff | none | 30 | 31 | True |
| push-back-v3 | k40 | k30 | 29 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | pure | k50 | 16 | 16 | True |
| reach-wall-v3 | k40 | k30 | 44 | 45 | True |
| shelf-place-v3 | pure | k50 | 16 | 16 | True |
| soccer-v3 | koff | none | 5 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 41 | 41 | True |
| sweep-into-v3 | koff | none | 31 | 21 | True |
| sweep-v3 | koff | none | 13 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k40 | k30 | 43 | 45 | True |
## π0.5 · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k40 | k30 | 21 | 19 | True |
| basketball-v3 | pure | k50 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 49 | True |
| button-press-v3 | k40 | k30 | 48 | 50 | True |
| button-press-wall-v3 | k50 | k40 | 50 | 50 | True |
| coffee-button-v3 | k40 | k30 | 48 | 49 | True |
| coffee-pull-v3 | koff | none | 40 | 40 | True |
| coffee-push-v3 | pure | k50 | 30 | 30 | True |
| dial-turn-v3 | koff | none | 1 | 1 | True |
| disassemble-v3 | koff | none | 31 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | pure | k50 | 49 | 49 | True |
| door-unlock-v3 | koff | none | 38 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k50 | k40 | 41 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k50 | k40 | 9 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | k20 | k10 | 15 | 16 | True |
| peg-insert-side-v3 | koff | none | 0 | 2 | True |
| peg-unplug-side-v3 | k50 | k40 | 17 | 18 | True |
| pick-out-of-hole-v3 | k40 | k30 | 10 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | koff | none | 18 | 7 | True |
| plate-slide-side-v3 | koff | none | 49 | 47 | True |
| plate-slide-v3 | koff | none | 33 | 31 | True |
| push-back-v3 | k30 | k20 | 29 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | k50 | k40 | 44 | 46 | True |
| reach-v3 | pure | k50 | 16 | 16 | True |
| reach-wall-v3 | k20 | k10 | 43 | 45 | True |
| shelf-place-v3 | k30 | k20 | 15 | 16 | True |
| soccer-v3 | koff | none | 5 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 47 | 41 | True |
| sweep-into-v3 | koff | none | 42 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k50 | k40 | 47 | 45 | True |
## π0.5 · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | koff | none | 29 | 19 | True |
| basketball-v3 | pure | k50 | 14 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | koff | none | 47 | 49 | True |
| coffee-pull-v3 | k20 | k10 | 39 | 40 | True |
| coffee-push-v3 | k50 | k40 | 31 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 30 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k50 | k40 | 12 | 14 | True |
| door-open-v3 | koff | none | 50 | 49 | True |
| door-unlock-v3 | koff | none | 48 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k50 | k40 | 43 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k50 | k40 | 9 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 1 | 0 | True |
| lever-pull-v3 | k20 | k10 | 18 | 16 | True |
| peg-insert-side-v3 | koff | none | 1 | 2 | True |
| peg-unplug-side-v3 | k50 | k40 | 17 | 18 | True |
| pick-out-of-hole-v3 | k20 | k10 | 10 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | koff | none | 16 | 7 | True |
| plate-slide-side-v3 | koff | none | 49 | 47 | True |
| plate-slide-v3 | koff | none | 33 | 31 | True |
| push-back-v3 | k10 | koff | 28 | 30 | True |
| push-v3 | k50 | k40 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | k30 | k20 | 14 | 16 | True |
| reach-wall-v3 | k30 | k20 | 45 | 45 | True |
| shelf-place-v3 | k20 | k10 | 14 | 16 | True |
| soccer-v3 | koff | none | 6 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 41 | True |
| sweep-into-v3 | koff | none | 26 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 45 | 45 | True |
## π0.5 · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | koff | none | 34 | 19 | True |
| basketball-v3 | k50 | k40 | 12 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | k10 | koff | 47 | 49 | True |
| coffee-pull-v3 | k20 | k10 | 40 | 40 | True |
| coffee-push-v3 | k30 | k20 | 29 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 29 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 14 | 14 | True |
| door-open-v3 | koff | none | 49 | 49 | True |
| door-unlock-v3 | koff | none | 49 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k30 | k20 | 42 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k50 | k40 | 10 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 14 | 16 | True |
| peg-insert-side-v3 | koff | none | 0 | 2 | True |
| peg-unplug-side-v3 | pure | k50 | 18 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 9 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | koff | none | 27 | 7 | True |
| plate-slide-side-v3 | koff | none | 49 | 47 | True |
| plate-slide-v3 | koff | none | 33 | 31 | True |
| push-back-v3 | koff | none | 29 | 30 | True |
| push-v3 | pure | k50 | 31 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | pure | k50 | 16 | 16 | True |
| reach-wall-v3 | koff | none | 45 | 45 | True |
| shelf-place-v3 | k20 | k10 | 15 | 16 | True |
| soccer-v3 | koff | none | 6 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 41 | True |
| sweep-into-v3 | koff | none | 30 | 21 | True |
| sweep-v3 | pure | k50 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 45 | 45 | True |
## π0.5 · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | koff | none | 31 | 19 | True |
| basketball-v3 | k30 | k20 | 16 | 14 | True |
| bin-picking-v3 | pure | k50 | 24 | 24 | True |
| box-close-v3 | pure | k50 | 18 | 18 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 49 | 49 | True |
| coffee-pull-v3 | k50 | k40 | 45 | 40 | True |
| coffee-push-v3 | k30 | k20 | 31 | 30 | True |
| dial-turn-v3 | koff | none | 0 | 1 | True |
| disassemble-v3 | koff | none | 31 | 22 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k40 | k30 | 15 | 14 | True |
| door-open-v3 | koff | none | 50 | 49 | True |
| door-unlock-v3 | koff | none | 49 | 38 | True |
| drawer-close-v3 | koff | none | 50 | 49 | True |
| drawer-open-v3 | k30 | k20 | 41 | 43 | True |
| faucet-close-v3 | pure | k50 | 26 | 26 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 9 | 9 | True |
| hand-insert-v3 | k50 | k40 | 10 | 11 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 23 | 16 | True |
| peg-insert-side-v3 | koff | none | 0 | 2 | True |
| peg-unplug-side-v3 | koff | none | 17 | 18 | True |
| pick-out-of-hole-v3 | k10 | koff | 6 | 8 | True |
| pick-place-v3 | pure | k50 | 17 | 17 | True |
| pick-place-wall-v3 | pure | k50 | 18 | 18 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 33 | True |
| plate-slide-back-v3 | koff | none | 29 | 7 | True |
| plate-slide-side-v3 | koff | none | 49 | 47 | True |
| plate-slide-v3 | koff | none | 34 | 31 | True |
| push-back-v3 | koff | none | 30 | 30 | True |
| push-v3 | k40 | k30 | 29 | 31 | True |
| push-wall-v3 | pure | k50 | 46 | 46 | True |
| reach-v3 | k30 | k20 | 14 | 16 | True |
| reach-wall-v3 | koff | none | 45 | 45 | True |
| shelf-place-v3 | k20 | k10 | 18 | 16 | True |
| soccer-v3 | koff | none | 5 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 41 | True |
| sweep-into-v3 | koff | none | 29 | 21 | True |
| sweep-v3 | k50 | k40 | 11 | 11 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 45 | 45 | True |
