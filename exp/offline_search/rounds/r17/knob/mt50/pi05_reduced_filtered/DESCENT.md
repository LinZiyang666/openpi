# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| π0.5 | 250 | 50/50 | 49.4% | 60.8% | 48.7% | 59.6% | 0.677 | False |
| π0.5 | 500 | 50/50 | 49.5% | 60.7% | 48.7% | 59.6% | 0.646 | False |
| π0.5 | 1000 | 50/50 | 51.1% | 60.4% | 48.7% | 59.6% | 0.618 | False |
| π0.5 | 1500 | 50/50 | 52.6% | 61.8% | 48.7% | 59.6% | 0.599 | False |
| π0.5 | 2000 | 50/50 | 53.5% | 62.6% | 48.7% | 59.6% | 0.606 | False |
| π0.5 | 2500 | 50/50 | 53.8% | 63.1% | 48.7% | 59.6% | 0.570 | False |

## π0.5 · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 15 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 50 | True |
| button-press-v3 | k50 | k40 | 50 | 50 | True |
| button-press-wall-v3 | k20 | k10 | 50 | 50 | True |
| coffee-button-v3 | pure | k50 | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 40 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 21 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | pure | k50 | 48 | 48 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 48 | 50 | True |
| faucet-close-v3 | k40 | k30 | 16 | 8 | True |
| faucet-open-v3 | k40 | k30 | 49 | 50 | True |
| hammer-v3 | koff | none | 12 | 13 | True |
| hand-insert-v3 | koff | none | 12 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 3 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | k20 | k10 | 9 | 6 | True |
| pick-out-of-hole-v3 | pure | k50 | 4 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | pure | k50 | 47 | 47 | True |
| plate-slide-back-v3 | koff | none | 41 | 16 | True |
| plate-slide-side-v3 | koff | none | 48 | 50 | True |
| plate-slide-v3 | pure | k50 | 38 | 38 | True |
| push-back-v3 | pure | k50 | 33 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | koff | none | 20 | 21 | True |
| reach-wall-v3 | pure | k50 | 43 | 43 | True |
| shelf-place-v3 | koff | none | 0 | 2 | True |
| soccer-v3 | koff | none | 1 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 45 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | koff | none | 19 | 18 | True |
| window-close-v3 | koff | none | 48 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 15 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | k10 | koff | 48 | 50 | True |
| button-press-topdown-wall-v3 | k10 | koff | 48 | 50 | True |
| button-press-v3 | pure | k50 | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | pure | k50 | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 40 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 24 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 47 | 48 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k50 | k40 | 49 | 50 | True |
| faucet-close-v3 | k50 | k40 | 11 | 8 | True |
| faucet-open-v3 | k30 | k20 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | k20 | k10 | 8 | 6 | True |
| pick-out-of-hole-v3 | pure | k50 | 4 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | koff | none | 43 | 16 | True |
| plate-slide-side-v3 | koff | none | 48 | 50 | True |
| plate-slide-v3 | k50 | k40 | 36 | 38 | True |
| push-back-v3 | k50 | k40 | 31 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | pure | k50 | 21 | 21 | True |
| reach-wall-v3 | koff | none | 43 | 43 | True |
| shelf-place-v3 | koff | none | 0 | 2 | True |
| soccer-v3 | koff | none | 0 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 45 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | koff | none | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 20 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | k50 | k40 | 15 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 50 | True |
| button-press-v3 | koff | none | 49 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | k30 | k20 | 49 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 42 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 31 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 49 | 48 | True |
| door-unlock-v3 | k40 | k30 | 46 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 8 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | k50 | k40 | 18 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 8 | 6 | True |
| pick-out-of-hole-v3 | pure | k50 | 4 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | koff | none | 18 | 16 | True |
| plate-slide-side-v3 | koff | none | 49 | 50 | True |
| plate-slide-v3 | k50 | k40 | 36 | 38 | True |
| push-back-v3 | k30 | k20 | 32 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | k50 | k40 | 46 | 48 | True |
| reach-v3 | pure | k50 | 21 | 21 | True |
| reach-wall-v3 | koff | none | 42 | 43 | True |
| shelf-place-v3 | koff | none | 8 | 2 | True |
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
| assembly-v3 | koff | none | 28 | 15 | True |
| basketball-v3 | pure | k50 | 19 | 19 | True |
| bin-picking-v3 | pure | k50 | 17 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k10 | koff | 50 | 50 | True |
| coffee-button-v3 | koff | none | 48 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 43 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 29 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 50 | 48 | True |
| door-unlock-v3 | koff | none | 49 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | k40 | k30 | 10 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | pure | k50 | 19 | 19 | True |
| peg-insert-side-v3 | k20 | k10 | 2 | 4 | True |
| peg-unplug-side-v3 | koff | none | 15 | 6 | True |
| pick-out-of-hole-v3 | k50 | k40 | 2 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | k40 | k30 | 16 | 16 | True |
| plate-slide-side-v3 | koff | none | 49 | 50 | True |
| plate-slide-v3 | k50 | k40 | 37 | 38 | True |
| push-back-v3 | k40 | k30 | 34 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | pure | k50 | 21 | 21 | True |
| reach-wall-v3 | koff | none | 44 | 43 | True |
| shelf-place-v3 | koff | none | 9 | 2 | True |
| soccer-v3 | koff | none | 7 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | koff | none | 33 | 15 | True |
| basketball-v3 | k40 | k30 | 18 | 19 | True |
| bin-picking-v3 | k50 | k40 | 15 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k10 | koff | 50 | 50 | True |
| coffee-button-v3 | k20 | k10 | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 41 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 31 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 20 | 20 | True |
| door-open-v3 | koff | none | 50 | 48 | True |
| door-unlock-v3 | koff | none | 49 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k50 | k40 | 48 | 50 | True |
| faucet-close-v3 | pure | k50 | 8 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | k30 | k20 | 35 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 12 | 6 | True |
| pick-out-of-hole-v3 | pure | k50 | 4 | 4 | True |
| pick-place-v3 | pure | k50 | 14 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | pure | k50 | 16 | 16 | True |
| plate-slide-side-v3 | koff | none | 49 | 50 | True |
| plate-slide-v3 | k50 | k40 | 37 | 38 | True |
| push-back-v3 | k30 | k20 | 31 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | pure | k50 | 21 | 21 | True |
| reach-wall-v3 | koff | none | 45 | 43 | True |
| shelf-place-v3 | koff | none | 13 | 2 | True |
| soccer-v3 | koff | none | 7 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
## π0.5 · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | koff | none | 31 | 15 | True |
| basketball-v3 | k40 | k30 | 29 | 19 | True |
| bin-picking-v3 | k40 | k30 | 15 | 17 | True |
| box-close-v3 | pure | k50 | 15 | 15 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 39 | 40 | True |
| coffee-push-v3 | pure | k50 | 46 | 46 | True |
| dial-turn-v3 | pure | k50 | 12 | 12 | True |
| disassemble-v3 | koff | none | 31 | 16 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | k50 | k40 | 22 | 20 | True |
| door-open-v3 | koff | none | 50 | 48 | True |
| door-unlock-v3 | koff | none | 49 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 50 | 50 | True |
| faucet-close-v3 | k40 | k30 | 11 | 8 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 13 | 13 | True |
| hand-insert-v3 | pure | k50 | 14 | 14 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 1 | True |
| handle-pull-v3 | koff | none | 0 | 0 | True |
| lever-pull-v3 | koff | none | 23 | 19 | True |
| peg-insert-side-v3 | pure | k50 | 4 | 4 | True |
| peg-unplug-side-v3 | koff | none | 14 | 6 | True |
| pick-out-of-hole-v3 | pure | k50 | 4 | 4 | True |
| pick-place-v3 | k50 | k40 | 12 | 14 | True |
| pick-place-wall-v3 | pure | k50 | 25 | 25 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 47 | True |
| plate-slide-back-v3 | koff | none | 28 | 16 | True |
| plate-slide-side-v3 | koff | none | 49 | 50 | True |
| plate-slide-v3 | k50 | k40 | 39 | 38 | True |
| push-back-v3 | koff | none | 33 | 33 | True |
| push-v3 | pure | k50 | 38 | 38 | True |
| push-wall-v3 | pure | k50 | 48 | 48 | True |
| reach-v3 | k50 | k40 | 19 | 21 | True |
| reach-wall-v3 | koff | none | 44 | 43 | True |
| shelf-place-v3 | koff | none | 12 | 2 | True |
| soccer-v3 | koff | none | 5 | 1 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 48 | 45 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 18 | 18 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 50 | 50 | True |
