# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GR00T | 250 | 50/50 | 66.0% | 74.6% | 66.3% | 75.2% | 0.657 | False |
| GR00T | 500 | 50/50 | 65.9% | 74.4% | 66.3% | 75.2% | 0.594 | False |
| GR00T | 1000 | 50/50 | 66.2% | 74.7% | 66.3% | 75.2% | 0.588 | False |
| GR00T | 1500 | 50/50 | 66.0% | 74.8% | 66.3% | 75.2% | 0.561 | False |
| GR00T | 2000 | 50/50 | 66.0% | 74.8% | 66.3% | 75.2% | 0.531 | False |
| GR00T | 2500 | 50/50 | 66.3% | 74.9% | 66.3% | 75.2% | 0.559 | False |

## GR00T · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 49 | 49 | True |
| basketball-v3 | pure | k50 | 25 | 25 | True |
| bin-picking-v3 | koff | none | 0 | 1 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 50 | True |
| button-press-v3 | k40 | k30 | 49 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | k30 | k20 | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 39 | 39 | True |
| coffee-push-v3 | pure | k50 | 44 | 44 | True |
| dial-turn-v3 | pure | k50 | 47 | 47 | True |
| disassemble-v3 | k20 | k10 | 24 | 21 | True |
| door-close-v3 | koff | none | 49 | 50 | True |
| door-lock-v3 | pure | k50 | 30 | 30 | True |
| door-open-v3 | k30 | k20 | 49 | 50 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | k10 | koff | 49 | 50 | True |
| drawer-open-v3 | k20 | k10 | 48 | 50 | True |
| faucet-close-v3 | pure | k50 | 42 | 42 | True |
| faucet-open-v3 | k30 | k20 | 49 | 50 | True |
| hammer-v3 | pure | k50 | 46 | 46 | True |
| hand-insert-v3 | pure | k50 | 32 | 32 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k40 | k30 | 48 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 6 | 6 | True |
| handle-pull-v3 | pure | k50 | 23 | 23 | True |
| lever-pull-v3 | pure | k50 | 34 | 34 | True |
| peg-insert-side-v3 | k50 | k40 | 17 | 19 | True |
| peg-unplug-side-v3 | pure | k50 | 34 | 34 | True |
| pick-out-of-hole-v3 | pure | k50 | 34 | 34 | True |
| pick-place-v3 | pure | k50 | 34 | 34 | True |
| pick-place-wall-v3 | pure | k50 | 41 | 41 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | pure | k50 | 46 | 46 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k50 | k40 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 43 | 43 | True |
| push-wall-v3 | pure | k50 | 38 | 38 | True |
| reach-v3 | pure | k50 | 33 | 33 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | pure | k50 | 24 | 24 | True |
| soccer-v3 | pure | k50 | 19 | 19 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k40 | k30 | 49 | 50 | True |
| sweep-into-v3 | pure | k50 | 38 | 38 | True |
| sweep-v3 | pure | k50 | 27 | 27 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 44 | 44 | True |
## GR00T · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 49 | 49 | True |
| basketball-v3 | pure | k50 | 25 | 25 | True |
| bin-picking-v3 | koff | none | 4 | 1 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | k30 | k20 | 48 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 50 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 48 | 50 | True |
| coffee-button-v3 | k20 | k10 | 48 | 50 | True |
| coffee-pull-v3 | pure | k50 | 39 | 39 | True |
| coffee-push-v3 | pure | k50 | 44 | 44 | True |
| dial-turn-v3 | pure | k50 | 47 | 47 | True |
| disassemble-v3 | koff | none | 23 | 21 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 30 | 30 | True |
| door-open-v3 | koff | none | 49 | 50 | True |
| door-unlock-v3 | k50 | k40 | 46 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k20 | k10 | 49 | 50 | True |
| faucet-close-v3 | pure | k50 | 42 | 42 | True |
| faucet-open-v3 | k40 | k30 | 49 | 50 | True |
| hammer-v3 | pure | k50 | 46 | 46 | True |
| hand-insert-v3 | k40 | k30 | 30 | 32 | True |
| handle-press-side-v3 | k20 | k10 | 48 | 50 | True |
| handle-press-v3 | k50 | k40 | 48 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 6 | 6 | True |
| handle-pull-v3 | pure | k50 | 23 | 23 | True |
| lever-pull-v3 | pure | k50 | 34 | 34 | True |
| peg-insert-side-v3 | k50 | k40 | 17 | 19 | True |
| peg-unplug-side-v3 | k40 | k30 | 38 | 34 | True |
| pick-out-of-hole-v3 | pure | k50 | 34 | 34 | True |
| pick-place-v3 | pure | k50 | 34 | 34 | True |
| pick-place-wall-v3 | pure | k50 | 41 | 41 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 45 | 46 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k30 | k20 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 43 | 43 | True |
| push-wall-v3 | pure | k50 | 38 | 38 | True |
| reach-v3 | k50 | k40 | 31 | 33 | True |
| reach-wall-v3 | k40 | k30 | 40 | 42 | True |
| shelf-place-v3 | k50 | k40 | 26 | 24 | True |
| soccer-v3 | pure | k50 | 19 | 19 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 49 | 50 | True |
| sweep-into-v3 | pure | k50 | 38 | 38 | True |
| sweep-v3 | pure | k50 | 27 | 27 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 44 | 44 | True |
## GR00T · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 49 | 49 | True |
| basketball-v3 | k50 | k40 | 25 | 25 | True |
| bin-picking-v3 | koff | none | 6 | 1 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | k10 | koff | 49 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 50 | True |
| button-press-v3 | k20 | k10 | 49 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 48 | 50 | True |
| coffee-pull-v3 | k30 | k20 | 38 | 39 | True |
| coffee-push-v3 | pure | k50 | 44 | 44 | True |
| dial-turn-v3 | pure | k50 | 47 | 47 | True |
| disassemble-v3 | koff | none | 26 | 21 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 30 | 30 | True |
| door-open-v3 | k20 | k10 | 48 | 50 | True |
| door-unlock-v3 | k50 | k40 | 46 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 42 | 42 | True |
| faucet-open-v3 | koff | none | 48 | 50 | True |
| hammer-v3 | pure | k50 | 46 | 46 | True |
| hand-insert-v3 | k30 | k20 | 30 | 32 | True |
| handle-press-side-v3 | koff | none | 49 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 6 | 6 | True |
| handle-pull-v3 | pure | k50 | 23 | 23 | True |
| lever-pull-v3 | pure | k50 | 34 | 34 | True |
| peg-insert-side-v3 | k50 | k40 | 18 | 19 | True |
| peg-unplug-side-v3 | pure | k50 | 34 | 34 | True |
| pick-out-of-hole-v3 | pure | k50 | 34 | 34 | True |
| pick-place-v3 | pure | k50 | 34 | 34 | True |
| pick-place-wall-v3 | pure | k50 | 41 | 41 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | k10 | koff | 47 | 46 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k10 | koff | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 43 | 43 | True |
| push-wall-v3 | pure | k50 | 38 | 38 | True |
| reach-v3 | pure | k50 | 33 | 33 | True |
| reach-wall-v3 | k40 | k30 | 41 | 42 | True |
| shelf-place-v3 | pure | k50 | 24 | 24 | True |
| soccer-v3 | pure | k50 | 19 | 19 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 49 | 50 | True |
| sweep-into-v3 | pure | k50 | 38 | 38 | True |
| sweep-v3 | pure | k50 | 27 | 27 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | pure | k50 | 44 | 44 | True |
## GR00T · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 49 | 49 | True |
| basketball-v3 | k50 | k40 | 24 | 25 | True |
| bin-picking-v3 | koff | none | 3 | 1 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | koff | none | 49 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 39 | 39 | True |
| coffee-push-v3 | pure | k50 | 44 | 44 | True |
| dial-turn-v3 | pure | k50 | 47 | 47 | True |
| disassemble-v3 | koff | none | 26 | 21 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 30 | 30 | True |
| door-open-v3 | k20 | k10 | 49 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 42 | 42 | True |
| faucet-open-v3 | k40 | k30 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 46 | 46 | True |
| hand-insert-v3 | k30 | k20 | 32 | 32 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 6 | 6 | True |
| handle-pull-v3 | pure | k50 | 23 | 23 | True |
| lever-pull-v3 | pure | k50 | 34 | 34 | True |
| peg-insert-side-v3 | k50 | k40 | 19 | 19 | True |
| peg-unplug-side-v3 | k50 | k40 | 36 | 34 | True |
| pick-out-of-hole-v3 | pure | k50 | 34 | 34 | True |
| pick-place-v3 | pure | k50 | 34 | 34 | True |
| pick-place-wall-v3 | k50 | k40 | 39 | 41 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 45 | 46 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k10 | koff | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | k40 | k30 | 41 | 43 | True |
| push-wall-v3 | pure | k50 | 38 | 38 | True |
| reach-v3 | k40 | k30 | 31 | 33 | True |
| reach-wall-v3 | k50 | k40 | 40 | 42 | True |
| shelf-place-v3 | pure | k50 | 24 | 24 | True |
| soccer-v3 | pure | k50 | 19 | 19 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 49 | 50 | True |
| sweep-into-v3 | pure | k50 | 38 | 38 | True |
| sweep-v3 | pure | k50 | 27 | 27 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k40 | k30 | 44 | 44 | True |
## GR00T · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k50 | k40 | 47 | 49 | True |
| basketball-v3 | pure | k50 | 25 | 25 | True |
| bin-picking-v3 | koff | none | 6 | 1 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 48 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k30 | k20 | 39 | 39 | True |
| coffee-push-v3 | pure | k50 | 44 | 44 | True |
| dial-turn-v3 | k40 | k30 | 46 | 47 | True |
| disassemble-v3 | koff | none | 26 | 21 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 30 | 30 | True |
| door-open-v3 | koff | none | 48 | 50 | True |
| door-unlock-v3 | koff | none | 49 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 42 | 42 | True |
| faucet-open-v3 | k40 | k30 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 46 | 46 | True |
| hand-insert-v3 | k30 | k20 | 31 | 32 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 6 | 6 | True |
| handle-pull-v3 | pure | k50 | 23 | 23 | True |
| lever-pull-v3 | pure | k50 | 34 | 34 | True |
| peg-insert-side-v3 | k40 | k30 | 19 | 19 | True |
| peg-unplug-side-v3 | k40 | k30 | 35 | 34 | True |
| pick-out-of-hole-v3 | pure | k50 | 34 | 34 | True |
| pick-place-v3 | pure | k50 | 34 | 34 | True |
| pick-place-wall-v3 | k50 | k40 | 39 | 41 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 45 | 46 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | koff | none | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | k40 | k30 | 41 | 43 | True |
| push-wall-v3 | pure | k50 | 38 | 38 | True |
| reach-v3 | k30 | k20 | 34 | 33 | True |
| reach-wall-v3 | k50 | k40 | 41 | 42 | True |
| shelf-place-v3 | pure | k50 | 24 | 24 | True |
| soccer-v3 | pure | k50 | 19 | 19 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 50 | 50 | True |
| sweep-into-v3 | pure | k50 | 38 | 38 | True |
| sweep-v3 | k50 | k40 | 25 | 27 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k40 | k30 | 44 | 44 | True |
## GR00T · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 49 | 49 | True |
| basketball-v3 | pure | k50 | 25 | 25 | True |
| bin-picking-v3 | koff | none | 3 | 1 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | k10 | koff | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 48 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k30 | k20 | 38 | 39 | True |
| coffee-push-v3 | pure | k50 | 44 | 44 | True |
| dial-turn-v3 | pure | k50 | 47 | 47 | True |
| disassemble-v3 | koff | none | 25 | 21 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 30 | 30 | True |
| door-open-v3 | koff | none | 49 | 50 | True |
| door-unlock-v3 | koff | none | 49 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 42 | 42 | True |
| faucet-open-v3 | k40 | k30 | 48 | 50 | True |
| hammer-v3 | pure | k50 | 46 | 46 | True |
| hand-insert-v3 | k30 | k20 | 33 | 32 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 6 | 6 | True |
| handle-pull-v3 | pure | k50 | 23 | 23 | True |
| lever-pull-v3 | pure | k50 | 34 | 34 | True |
| peg-insert-side-v3 | k50 | k40 | 19 | 19 | True |
| peg-unplug-side-v3 | pure | k50 | 34 | 34 | True |
| pick-out-of-hole-v3 | pure | k50 | 34 | 34 | True |
| pick-place-v3 | pure | k50 | 34 | 34 | True |
| pick-place-wall-v3 | pure | k50 | 41 | 41 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | k30 | k20 | 46 | 46 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | koff | none | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 2 | True |
| push-v3 | pure | k50 | 43 | 43 | True |
| push-wall-v3 | pure | k50 | 38 | 38 | True |
| reach-v3 | k20 | k10 | 31 | 33 | True |
| reach-wall-v3 | k50 | k40 | 41 | 42 | True |
| shelf-place-v3 | k40 | k30 | 22 | 24 | True |
| soccer-v3 | pure | k50 | 19 | 19 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 50 | 50 | True |
| sweep-into-v3 | pure | k50 | 38 | 38 | True |
| sweep-v3 | k20 | k10 | 26 | 27 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k40 | k30 | 45 | 44 | True |
