# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GR00T | 250 | 50/50 | 66.5% | 75.3% | 66.8% | 75.6% | 0.739 | False |
| GR00T | 500 | 50/50 | 67.4% | 75.7% | 66.8% | 75.6% | 0.698 | False |
| GR00T | 1000 | 50/50 | 67.4% | 75.6% | 66.8% | 75.6% | 0.673 | False |
| GR00T | 1500 | 50/50 | 68.0% | 76.4% | 66.8% | 75.6% | 0.643 | False |
| GR00T | 2000 | 50/50 | 67.2% | 75.6% | 66.8% | 75.6% | 0.599 | False |
| GR00T | 2500 | 50/50 | 67.1% | 75.6% | 66.8% | 75.6% | 0.604 | False |

## GR00T · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 50 | 50 | True |
| basketball-v3 | pure | k50 | 23 | 23 | True |
| bin-picking-v3 | k20 | k10 | 8 | 5 | True |
| box-close-v3 | pure | k50 | 28 | 28 | True |
| button-press-topdown-v3 | k30 | k20 | 49 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 49 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | k30 | k20 | 50 | 50 | True |
| coffee-pull-v3 | k30 | k20 | 21 | 20 | True |
| coffee-push-v3 | k40 | k30 | 33 | 35 | True |
| dial-turn-v3 | pure | k50 | 46 | 46 | True |
| disassemble-v3 | k10 | koff | 13 | 15 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | k30 | k20 | 50 | 50 | True |
| door-unlock-v3 | pure | k50 | 49 | 49 | True |
| drawer-close-v3 | k10 | koff | 48 | 50 | True |
| drawer-open-v3 | k20 | k10 | 49 | 50 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | k30 | k20 | 49 | 50 | True |
| hammer-v3 | pure | k50 | 49 | 49 | True |
| hand-insert-v3 | pure | k50 | 38 | 38 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k50 | k40 | 48 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 8 | 8 | True |
| handle-pull-v3 | pure | k50 | 31 | 31 | True |
| lever-pull-v3 | pure | k50 | 37 | 37 | True |
| peg-insert-side-v3 | k50 | k40 | 17 | 17 | True |
| peg-unplug-side-v3 | k50 | k40 | 31 | 31 | True |
| pick-out-of-hole-v3 | pure | k50 | 30 | 30 | True |
| pick-place-v3 | pure | k50 | 36 | 36 | True |
| pick-place-wall-v3 | pure | k50 | 46 | 46 | True |
| plate-slide-back-side-v3 | pure | k50 | 50 | 50 | True |
| plate-slide-back-v3 | k30 | k20 | 45 | 45 | True |
| plate-slide-side-v3 | koff | none | 50 | 48 | True |
| plate-slide-v3 | k50 | k40 | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 45 | 45 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | pure | k50 | 40 | 40 | True |
| reach-wall-v3 | pure | k50 | 43 | 43 | True |
| shelf-place-v3 | pure | k50 | 31 | 31 | True |
| soccer-v3 | pure | k50 | 21 | 21 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 50 | 50 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 28 | 28 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 46 | 46 | True |
## GR00T · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 50 | 50 | True |
| basketball-v3 | pure | k50 | 23 | 23 | True |
| bin-picking-v3 | koff | none | 4 | 5 | True |
| box-close-v3 | pure | k50 | 28 | 28 | True |
| button-press-topdown-v3 | k30 | k20 | 49 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 49 | 49 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | k20 | k10 | 49 | 50 | True |
| coffee-button-v3 | k20 | k10 | 49 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 24 | 20 | True |
| coffee-push-v3 | k50 | k40 | 33 | 35 | True |
| dial-turn-v3 | pure | k50 | 46 | 46 | True |
| disassemble-v3 | koff | none | 24 | 15 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | koff | none | 50 | 50 | True |
| door-unlock-v3 | pure | k50 | 49 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k20 | k10 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | k40 | k30 | 49 | 50 | True |
| hammer-v3 | pure | k50 | 49 | 49 | True |
| hand-insert-v3 | pure | k50 | 38 | 38 | True |
| handle-press-side-v3 | k30 | k20 | 50 | 50 | True |
| handle-press-v3 | pure | k50 | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 8 | 8 | True |
| handle-pull-v3 | pure | k50 | 31 | 31 | True |
| lever-pull-v3 | pure | k50 | 37 | 37 | True |
| peg-insert-side-v3 | k40 | k30 | 16 | 17 | True |
| peg-unplug-side-v3 | k20 | k10 | 34 | 31 | True |
| pick-out-of-hole-v3 | pure | k50 | 30 | 30 | True |
| pick-place-v3 | pure | k50 | 36 | 36 | True |
| pick-place-wall-v3 | pure | k50 | 46 | 46 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 44 | 45 | True |
| plate-slide-side-v3 | koff | none | 50 | 48 | True |
| plate-slide-v3 | k30 | k20 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 45 | 45 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | pure | k50 | 40 | 40 | True |
| reach-wall-v3 | k40 | k30 | 41 | 43 | True |
| shelf-place-v3 | pure | k50 | 31 | 31 | True |
| soccer-v3 | pure | k50 | 21 | 21 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 49 | 50 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 28 | 28 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 46 | 46 | True |
## GR00T · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 50 | 50 | True |
| basketball-v3 | k40 | k30 | 23 | 23 | True |
| bin-picking-v3 | k20 | k10 | 6 | 5 | True |
| box-close-v3 | pure | k50 | 28 | 28 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 49 | True |
| button-press-v3 | k20 | k10 | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 48 | 50 | True |
| coffee-pull-v3 | koff | none | 22 | 20 | True |
| coffee-push-v3 | pure | k50 | 35 | 35 | True |
| dial-turn-v3 | pure | k50 | 46 | 46 | True |
| disassemble-v3 | koff | none | 26 | 15 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | k30 | k20 | 49 | 50 | True |
| door-unlock-v3 | k50 | k40 | 47 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | koff | none | 48 | 50 | True |
| hammer-v3 | pure | k50 | 49 | 49 | True |
| hand-insert-v3 | pure | k50 | 38 | 38 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 8 | 8 | True |
| handle-pull-v3 | pure | k50 | 31 | 31 | True |
| lever-pull-v3 | pure | k50 | 37 | 37 | True |
| peg-insert-side-v3 | k40 | k30 | 17 | 17 | True |
| peg-unplug-side-v3 | k40 | k30 | 30 | 31 | True |
| pick-out-of-hole-v3 | pure | k50 | 30 | 30 | True |
| pick-place-v3 | pure | k50 | 36 | 36 | True |
| pick-place-wall-v3 | k50 | k40 | 44 | 46 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | k10 | koff | 46 | 45 | True |
| plate-slide-side-v3 | koff | none | 50 | 48 | True |
| plate-slide-v3 | k10 | koff | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 45 | 45 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | pure | k50 | 40 | 40 | True |
| reach-wall-v3 | k40 | k30 | 42 | 43 | True |
| shelf-place-v3 | pure | k50 | 31 | 31 | True |
| soccer-v3 | pure | k50 | 21 | 21 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 49 | 50 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | pure | k50 | 28 | 28 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 46 | 46 | True |
## GR00T · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 50 | 50 | True |
| basketball-v3 | k50 | k40 | 22 | 23 | True |
| bin-picking-v3 | k20 | k10 | 6 | 5 | True |
| box-close-v3 | pure | k50 | 28 | 28 | True |
| button-press-topdown-v3 | koff | none | 49 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 22 | 20 | True |
| coffee-push-v3 | pure | k50 | 35 | 35 | True |
| dial-turn-v3 | pure | k50 | 46 | 46 | True |
| disassemble-v3 | koff | none | 26 | 15 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | k10 | koff | 49 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | koff | none | 49 | 50 | True |
| hammer-v3 | pure | k50 | 49 | 49 | True |
| hand-insert-v3 | pure | k50 | 38 | 38 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 8 | 8 | True |
| handle-pull-v3 | pure | k50 | 31 | 31 | True |
| lever-pull-v3 | pure | k50 | 37 | 37 | True |
| peg-insert-side-v3 | k40 | k30 | 20 | 17 | True |
| peg-unplug-side-v3 | k20 | k10 | 42 | 31 | True |
| pick-out-of-hole-v3 | pure | k50 | 30 | 30 | True |
| pick-place-v3 | pure | k50 | 36 | 36 | True |
| pick-place-wall-v3 | pure | k50 | 46 | 46 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 45 | 45 | True |
| plate-slide-side-v3 | koff | none | 50 | 48 | True |
| plate-slide-v3 | k20 | k10 | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | k50 | k40 | 43 | 45 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | pure | k50 | 40 | 40 | True |
| reach-wall-v3 | k50 | k40 | 44 | 43 | True |
| shelf-place-v3 | pure | k50 | 31 | 31 | True |
| soccer-v3 | k50 | k40 | 20 | 21 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 50 | 50 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | k50 | k40 | 31 | 28 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 46 | 46 | True |
## GR00T · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 49 | 50 | True |
| basketball-v3 | k50 | k40 | 22 | 23 | True |
| bin-picking-v3 | koff | none | 6 | 5 | True |
| box-close-v3 | pure | k50 | 28 | 28 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 49 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 21 | 20 | True |
| coffee-push-v3 | k40 | k30 | 33 | 35 | True |
| dial-turn-v3 | pure | k50 | 46 | 46 | True |
| disassemble-v3 | koff | none | 26 | 15 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | koff | none | 49 | 50 | True |
| door-unlock-v3 | pure | k50 | 49 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | k40 | k30 | 48 | 50 | True |
| hammer-v3 | pure | k50 | 49 | 49 | True |
| hand-insert-v3 | k40 | k30 | 36 | 38 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 8 | 8 | True |
| handle-pull-v3 | pure | k50 | 31 | 31 | True |
| lever-pull-v3 | pure | k50 | 37 | 37 | True |
| peg-insert-side-v3 | k20 | k10 | 16 | 17 | True |
| peg-unplug-side-v3 | k20 | k10 | 36 | 31 | True |
| pick-out-of-hole-v3 | pure | k50 | 30 | 30 | True |
| pick-place-v3 | k50 | k40 | 36 | 36 | True |
| pick-place-wall-v3 | k50 | k40 | 44 | 46 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 46 | 45 | True |
| plate-slide-side-v3 | koff | none | 50 | 48 | True |
| plate-slide-v3 | k10 | koff | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 45 | 45 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | pure | k50 | 40 | 40 | True |
| reach-wall-v3 | k40 | k30 | 41 | 43 | True |
| shelf-place-v3 | pure | k50 | 31 | 31 | True |
| soccer-v3 | k50 | k40 | 19 | 21 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 50 | 50 | True |
| sweep-into-v3 | k30 | k20 | 36 | 37 | True |
| sweep-v3 | k40 | k30 | 30 | 28 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 46 | 46 | True |
## GR00T · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k40 | k30 | 49 | 50 | True |
| basketball-v3 | k40 | k30 | 23 | 23 | True |
| bin-picking-v3 | koff | none | 3 | 5 | True |
| box-close-v3 | pure | k50 | 28 | 28 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | k10 | koff | 49 | 49 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 21 | 20 | True |
| coffee-push-v3 | pure | k50 | 35 | 35 | True |
| dial-turn-v3 | k40 | k30 | 45 | 46 | True |
| disassemble-v3 | koff | none | 24 | 15 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | koff | none | 49 | 50 | True |
| door-unlock-v3 | koff | none | 49 | 49 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | k50 | k40 | 49 | 50 | True |
| hammer-v3 | pure | k50 | 49 | 49 | True |
| hand-insert-v3 | k30 | k20 | 37 | 38 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 8 | 8 | True |
| handle-pull-v3 | pure | k50 | 31 | 31 | True |
| lever-pull-v3 | pure | k50 | 37 | 37 | True |
| peg-insert-side-v3 | k30 | k20 | 17 | 17 | True |
| peg-unplug-side-v3 | koff | none | 30 | 31 | True |
| pick-out-of-hole-v3 | pure | k50 | 30 | 30 | True |
| pick-place-v3 | k40 | k30 | 35 | 36 | True |
| pick-place-wall-v3 | k50 | k40 | 45 | 46 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | k10 | koff | 47 | 45 | True |
| plate-slide-side-v3 | koff | none | 50 | 48 | True |
| plate-slide-v3 | koff | none | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 45 | 45 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | pure | k50 | 40 | 40 | True |
| reach-wall-v3 | k40 | k30 | 43 | 43 | True |
| shelf-place-v3 | pure | k50 | 31 | 31 | True |
| soccer-v3 | pure | k50 | 21 | 21 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 50 | 50 | True |
| sweep-into-v3 | pure | k50 | 37 | 37 | True |
| sweep-v3 | k20 | k10 | 26 | 28 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 46 | 46 | True |
