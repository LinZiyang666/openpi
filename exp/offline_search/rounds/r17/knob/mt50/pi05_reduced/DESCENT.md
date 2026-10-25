# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| π0.5 | 250 | 50/50 | 49.4% | 59.4% | 48.7% | 59.6% | 0.425 | False |
| π0.5 | 500 | 50/50 | 50.4% | 59.9% | 48.7% | 59.6% | 0.393 | False |
| π0.5 | 1000 | 50/50 | 51.1% | 60.5% | 48.7% | 59.6% | 0.392 | False |
| π0.5 | 1500 | 50/50 | 51.8% | 61.1% | 48.7% | 59.6% | 0.384 | False |
| π0.5 | 2000 | 50/50 | 51.7% | 61.0% | 48.7% | 59.6% | 0.366 | False |
| π0.5 | 2500 | 50/50 | 52.5% | 61.4% | 48.7% | 59.6% | 0.351 | False |

## π0.5 · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 20 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | k50 | k40 | 18 | 15 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k20 | k10 | 48 | 50 | True |
| button-press-v3 | k50 | k40 | 50 | 50 | True |
| button-press-wall-v3 | k20 | k10 | 50 | 50 | True |
| coffee-button-v3 | pure | k50 | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 40 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 20 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | pure | k50 | 48 | 48 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 49 | 50 | True |
| faucet-close-v3 | k30 | k20 | 6 | 8 | True |
| faucet-open-v3 | k20 | k10 | 48 | 50 | True |
| hammer-v3 | k50 | k40 | 11 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 6 | 6 | True |
| pick-out-of-hole-v3 | k50 | k40 | 4 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | pure | k50 | 47 | 47 | True |
| plate-slide-back-v3 | k40 | k30 | 15 | 16 | True |
| plate-slide-side-v3 | koff | none | 48 | 50 | True |
| plate-slide-v3 | k50 | k40 | 36 | 38 | True |
| push-back-v3 | pure | k50 | 33 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | k50 | k40 | 20 | 21 | True |
| reach-wall-v3 | pure | k50 | 43 | 43 | True |
| shelf-place-v3 | koff | none | 3 | 2 | True |
| soccer-v3 | koff | none | 1 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 45 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 48 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 21 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | k50 | k40 | 21 | 15 | True |
| button-press-topdown-v3 | k10 | koff | 48 | 50 | True |
| button-press-topdown-wall-v3 | k10 | koff | 48 | 50 | True |
| button-press-v3 | k50 | k40 | 48 | 50 | True |
| button-press-wall-v3 | koff | none | 49 | 50 | True |
| coffee-button-v3 | pure | k50 | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 40 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 23 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 49 | 48 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 50 | 50 | True |
| faucet-close-v3 | k10 | koff | 6 | 8 | True |
| faucet-open-v3 | k30 | k20 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | k50 | k40 | 13 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 5 | 6 | True |
| pick-out-of-hole-v3 | k50 | k40 | 5 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | k50 | k40 | 14 | 16 | True |
| plate-slide-side-v3 | koff | none | 48 | 50 | True |
| plate-slide-v3 | k50 | k40 | 36 | 38 | True |
| push-back-v3 | k50 | k40 | 34 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | k40 | k30 | 19 | 21 | True |
| reach-wall-v3 | koff | none | 43 | 43 | True |
| shelf-place-v3 | koff | none | 6 | 2 | True |
| soccer-v3 | koff | none | 1 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 46 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k50 | k40 | 48 | 50 | True |
## π0.5 · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 24 | 15 | True |
| basketball-v3 | k50 | k40 | 17 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | k50 | k40 | 14 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 50 | True |
| button-press-v3 | k20 | k10 | 48 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | k30 | k20 | 49 | 50 | True |
| coffee-pull-v3 | k50 | k40 | 39 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 28 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 49 | 48 | True |
| door-unlock-v3 | k40 | k30 | 46 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 8 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | k30 | k20 | 12 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 18 | 6 | True |
| pick-out-of-hole-v3 | k40 | k30 | 2 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | pure | k50 | 16 | 16 | True |
| plate-slide-side-v3 | koff | none | 49 | 50 | True |
| plate-slide-v3 | k50 | k40 | 36 | 38 | True |
| push-back-v3 | k40 | k30 | 33 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | k20 | k10 | 19 | 21 | True |
| reach-wall-v3 | koff | none | 41 | 43 | True |
| shelf-place-v3 | koff | none | 10 | 2 | True |
| soccer-v3 | koff | none | 1 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 47 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 20 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | koff | none | 48 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 42 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 27 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 50 | 48 | True |
| door-unlock-v3 | koff | none | 50 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 8 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | k30 | k20 | 12 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 12 | 6 | True |
| pick-out-of-hole-v3 | koff | none | 9 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | pure | k50 | 16 | 16 | True |
| plate-slide-side-v3 | koff | none | 49 | 50 | True |
| plate-slide-v3 | k30 | k20 | 38 | 38 | True |
| push-back-v3 | koff | none | 31 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | k10 | koff | 19 | 21 | True |
| reach-wall-v3 | koff | none | 42 | 43 | True |
| shelf-place-v3 | koff | none | 12 | 2 | True |
| soccer-v3 | koff | none | 1 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 25 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k10 | koff | 49 | 50 | True |
| coffee-button-v3 | k20 | k10 | 50 | 50 | True |
| coffee-pull-v3 | k50 | k40 | 38 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 27 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k50 | k40 | 19 | 20 | True |
| door-open-v3 | koff | none | 50 | 48 | True |
| door-unlock-v3 | koff | none | 48 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 8 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | k20 | k10 | 22 | 19 | True |
| peg-insert-side-v3 | koff | none | 2 | 4 | True |
| peg-unplug-side-v3 | koff | none | 14 | 6 | True |
| pick-out-of-hole-v3 | k40 | k30 | 3 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | pure | k50 | 16 | 16 | True |
| plate-slide-side-v3 | koff | none | 48 | 50 | True |
| plate-slide-v3 | k30 | k20 | 38 | 38 | True |
| push-back-v3 | koff | none | 32 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | k10 | koff | 19 | 21 | True |
| reach-wall-v3 | koff | none | 43 | 43 | True |
| shelf-place-v3 | koff | none | 13 | 2 | True |
| soccer-v3 | koff | none | 0 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 45 | True |
| sweep-into-v3 | k50 | k40 | 35 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 24 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k50 | k40 | 39 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 30 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k50 | k40 | 19 | 20 | True |
| door-open-v3 | koff | none | 50 | 48 | True |
| door-unlock-v3 | koff | none | 49 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 48 | 50 | True |
| faucet-close-v3 | pure | k50 | 8 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | k50 | k40 | 15 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 20 | 19 | True |
| peg-insert-side-v3 | k20 | k10 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 14 | 6 | True |
| pick-out-of-hole-v3 | k30 | k20 | 4 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | pure | k50 | 16 | 16 | True |
| plate-slide-side-v3 | koff | none | 48 | 50 | True |
| plate-slide-v3 | k30 | k20 | 38 | 38 | True |
| push-back-v3 | koff | none | 35 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | k50 | k40 | 46 | 48 | True |
| reach-v3 | k10 | koff | 21 | 21 | True |
| reach-wall-v3 | koff | none | 43 | 43 | True |
| shelf-place-v3 | koff | none | 14 | 2 | True |
| soccer-v3 | koff | none | 1 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 45 | True |
| sweep-into-v3 | k50 | k40 | 35 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
