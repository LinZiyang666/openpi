# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DP | 250 | 50/50 | 46.4% | 61.8% | 45.6% | 61.0% | 0.564 | False |
| DP | 500 | 50/50 | 47.0% | 62.3% | 45.6% | 61.0% | 0.511 | False |
| DP | 1000 | 50/50 | 47.8% | 62.9% | 45.6% | 61.0% | 0.386 | False |
| DP | 1500 | 50/50 | 48.4% | 63.0% | 45.6% | 61.0% | 0.361 | False |
| DP | 2000 | 50/50 | 47.9% | 63.0% | 45.6% | 61.0% | 0.334 | False |
| DP | 2500 | 50/50 | 47.8% | 63.2% | 45.6% | 61.0% | 0.341 | False |

## DP · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 9 | 9 | True |
| basketball-v3 | pure | k50 | 11 | 11 | True |
| bin-picking-v3 | k30 | k20 | 21 | 22 | True |
| box-close-v3 | pure | k50 | 10 | 10 | True |
| button-press-topdown-v3 | k30 | k20 | 48 | 50 | True |
| button-press-topdown-wall-v3 | k40 | k30 | 50 | 50 | True |
| button-press-v3 | k40 | k30 | 49 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 48 | 50 | True |
| coffee-button-v3 | k30 | k20 | 50 | 50 | True |
| coffee-pull-v3 | pure | k50 | 34 | 34 | True |
| coffee-push-v3 | k50 | k40 | 31 | 30 | True |
| dial-turn-v3 | koff | none | 33 | 13 | True |
| disassemble-v3 | k30 | k20 | 14 | 9 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | pure | k50 | 50 | 50 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | k20 | k10 | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | k20 | k10 | 47 | 47 | True |
| faucet-open-v3 | k30 | k20 | 50 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 22 | 22 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | pure | k50 | 11 | 11 | True |
| lever-pull-v3 | k20 | k10 | 13 | 13 | True |
| peg-insert-side-v3 | pure | k50 | 26 | 26 | True |
| peg-unplug-side-v3 | pure | k50 | 34 | 34 | True |
| pick-out-of-hole-v3 | k10 | koff | 8 | 5 | True |
| pick-place-v3 | k40 | k30 | 16 | 16 | True |
| pick-place-wall-v3 | pure | k50 | 26 | 26 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 50 | 48 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | pure | k50 | 50 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | pure | k50 | 33 | 33 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | koff | none | 11 | 13 | True |
| reach-wall-v3 | k30 | k20 | 19 | 19 | True |
| shelf-place-v3 | pure | k50 | 8 | 8 | True |
| soccer-v3 | pure | k50 | 10 | 10 | True |
| stick-pull-v3 | koff | none | 0 | 2 | True |
| stick-push-v3 | pure | k50 | 31 | 31 | True |
| sweep-into-v3 | koff | none | 7 | 6 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 49 | 50 | True |
## DP · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 9 | 9 | True |
| basketball-v3 | pure | k50 | 11 | 11 | True |
| bin-picking-v3 | pure | k50 | 22 | 22 | True |
| box-close-v3 | k20 | k10 | 12 | 10 | True |
| button-press-topdown-v3 | k40 | k30 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 50 | True |
| button-press-v3 | k40 | k30 | 49 | 50 | True |
| button-press-wall-v3 | k40 | k30 | 50 | 50 | True |
| coffee-button-v3 | k20 | k10 | 48 | 50 | True |
| coffee-pull-v3 | pure | k50 | 34 | 34 | True |
| coffee-push-v3 | k30 | k20 | 35 | 30 | True |
| dial-turn-v3 | koff | none | 29 | 13 | True |
| disassemble-v3 | k20 | k10 | 13 | 9 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | koff | none | 48 | 50 | True |
| door-unlock-v3 | pure | k50 | 48 | 48 | True |
| drawer-close-v3 | k20 | k10 | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 47 | 47 | True |
| faucet-open-v3 | k20 | k10 | 50 | 50 | True |
| hammer-v3 | k40 | k30 | 30 | 31 | True |
| hand-insert-v3 | pure | k50 | 22 | 22 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 48 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | pure | k50 | 11 | 11 | True |
| lever-pull-v3 | k20 | k10 | 11 | 13 | True |
| peg-insert-side-v3 | pure | k50 | 26 | 26 | True |
| peg-unplug-side-v3 | k50 | k40 | 34 | 34 | True |
| pick-out-of-hole-v3 | k20 | k10 | 7 | 5 | True |
| pick-place-v3 | k40 | k30 | 16 | 16 | True |
| pick-place-wall-v3 | pure | k50 | 26 | 26 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | k20 | k10 | 48 | 48 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k30 | k20 | 49 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | pure | k50 | 33 | 33 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | koff | none | 12 | 13 | True |
| reach-wall-v3 | koff | none | 24 | 19 | True |
| shelf-place-v3 | pure | k50 | 8 | 8 | True |
| soccer-v3 | k50 | k40 | 9 | 10 | True |
| stick-pull-v3 | koff | none | 1 | 2 | True |
| stick-push-v3 | pure | k50 | 31 | 31 | True |
| sweep-into-v3 | koff | none | 19 | 6 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 50 | 50 | True |
## DP · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 9 | 9 | True |
| basketball-v3 | k30 | k20 | 10 | 11 | True |
| bin-picking-v3 | k30 | k20 | 27 | 22 | True |
| box-close-v3 | k20 | k10 | 13 | 10 | True |
| button-press-topdown-v3 | k30 | k20 | 49 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 50 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 49 | 50 | True |
| coffee-button-v3 | k20 | k10 | 50 | 50 | True |
| coffee-pull-v3 | k30 | k20 | 33 | 34 | True |
| coffee-push-v3 | koff | none | 31 | 30 | True |
| dial-turn-v3 | koff | none | 33 | 13 | True |
| disassemble-v3 | k20 | k10 | 11 | 9 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | koff | none | 50 | 50 | True |
| door-unlock-v3 | k30 | k20 | 48 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | k50 | k40 | 46 | 47 | True |
| faucet-open-v3 | koff | none | 49 | 50 | True |
| hammer-v3 | k30 | k20 | 29 | 31 | True |
| hand-insert-v3 | pure | k50 | 22 | 22 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | pure | k50 | 11 | 11 | True |
| lever-pull-v3 | koff | none | 14 | 13 | True |
| peg-insert-side-v3 | pure | k50 | 26 | 26 | True |
| peg-unplug-side-v3 | k40 | k30 | 33 | 34 | True |
| pick-out-of-hole-v3 | k10 | koff | 9 | 5 | True |
| pick-place-v3 | k40 | k30 | 17 | 16 | True |
| pick-place-wall-v3 | k30 | k20 | 29 | 26 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 49 | 48 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k50 | k40 | 32 | 33 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | koff | none | 13 | 13 | True |
| reach-wall-v3 | k40 | k30 | 19 | 19 | True |
| shelf-place-v3 | k10 | koff | 6 | 8 | True |
| soccer-v3 | pure | k50 | 10 | 10 | True |
| stick-pull-v3 | koff | none | 5 | 2 | True |
| stick-push-v3 | pure | k50 | 31 | 31 | True |
| sweep-into-v3 | koff | none | 25 | 6 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 49 | 50 | True |
## DP · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 9 | 9 | True |
| basketball-v3 | koff | none | 11 | 11 | True |
| bin-picking-v3 | koff | none | 22 | 22 | True |
| box-close-v3 | k20 | k10 | 17 | 10 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 36 | 34 | True |
| coffee-push-v3 | koff | none | 39 | 30 | True |
| dial-turn-v3 | koff | none | 18 | 13 | True |
| disassemble-v3 | k10 | koff | 8 | 9 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | k10 | koff | 49 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | pure | k50 | 47 | 47 | True |
| faucet-open-v3 | k40 | k30 | 49 | 50 | True |
| hammer-v3 | k30 | k20 | 29 | 31 | True |
| hand-insert-v3 | pure | k50 | 22 | 22 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | pure | k50 | 11 | 11 | True |
| lever-pull-v3 | koff | none | 20 | 13 | True |
| peg-insert-side-v3 | k40 | k30 | 27 | 26 | True |
| peg-unplug-side-v3 | k40 | k30 | 35 | 34 | True |
| pick-out-of-hole-v3 | k10 | koff | 14 | 5 | True |
| pick-place-v3 | koff | none | 15 | 16 | True |
| pick-place-wall-v3 | k30 | k20 | 30 | 26 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | k10 | koff | 50 | 48 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | pure | k50 | 33 | 33 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | koff | none | 12 | 13 | True |
| reach-wall-v3 | k40 | k30 | 18 | 19 | True |
| shelf-place-v3 | pure | k50 | 8 | 8 | True |
| soccer-v3 | k30 | k20 | 9 | 10 | True |
| stick-pull-v3 | koff | none | 8 | 2 | True |
| stick-push-v3 | pure | k50 | 31 | 31 | True |
| sweep-into-v3 | koff | none | 14 | 6 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 50 | 50 | True |
## DP · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 9 | 9 | True |
| basketball-v3 | koff | none | 13 | 11 | True |
| bin-picking-v3 | koff | none | 22 | 22 | True |
| box-close-v3 | k20 | k10 | 17 | 10 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 33 | 34 | True |
| coffee-push-v3 | koff | none | 40 | 30 | True |
| dial-turn-v3 | koff | none | 21 | 13 | True |
| disassemble-v3 | k10 | koff | 9 | 9 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | koff | none | 49 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | k40 | k30 | 45 | 47 | True |
| faucet-open-v3 | koff | none | 48 | 50 | True |
| hammer-v3 | k30 | k20 | 29 | 31 | True |
| hand-insert-v3 | pure | k50 | 22 | 22 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | pure | k50 | 11 | 11 | True |
| lever-pull-v3 | koff | none | 21 | 13 | True |
| peg-insert-side-v3 | k40 | k30 | 29 | 26 | True |
| peg-unplug-side-v3 | k40 | k30 | 35 | 34 | True |
| pick-out-of-hole-v3 | k30 | k20 | 9 | 5 | True |
| pick-place-v3 | k30 | k20 | 15 | 16 | True |
| pick-place-wall-v3 | pure | k50 | 26 | 26 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 50 | 48 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 50 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 31 | 33 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | koff | none | 12 | 13 | True |
| reach-wall-v3 | k30 | k20 | 18 | 19 | True |
| shelf-place-v3 | koff | none | 7 | 8 | True |
| soccer-v3 | k50 | k40 | 8 | 10 | True |
| stick-pull-v3 | koff | none | 9 | 2 | True |
| stick-push-v3 | pure | k50 | 31 | 31 | True |
| sweep-into-v3 | koff | none | 21 | 6 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 50 | 50 | True |
## DP · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 9 | 9 | True |
| basketball-v3 | koff | none | 15 | 11 | True |
| bin-picking-v3 | koff | none | 24 | 22 | True |
| box-close-v3 | k20 | k10 | 17 | 10 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 34 | 34 | True |
| coffee-push-v3 | koff | none | 39 | 30 | True |
| dial-turn-v3 | koff | none | 25 | 13 | True |
| disassemble-v3 | k10 | koff | 8 | 9 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 31 | 31 | True |
| door-open-v3 | k10 | koff | 48 | 50 | True |
| door-unlock-v3 | koff | none | 47 | 48 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | pure | k50 | 50 | 50 | True |
| faucet-close-v3 | k50 | k40 | 48 | 47 | True |
| faucet-open-v3 | k40 | k30 | 49 | 50 | True |
| hammer-v3 | k30 | k20 | 29 | 31 | True |
| hand-insert-v3 | pure | k50 | 22 | 22 | True |
| handle-press-side-v3 | koff | none | 50 | 50 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | koff | none | 0 | 0 | True |
| handle-pull-v3 | pure | k50 | 11 | 11 | True |
| lever-pull-v3 | koff | none | 26 | 13 | True |
| peg-insert-side-v3 | k30 | k20 | 24 | 26 | True |
| peg-unplug-side-v3 | koff | none | 32 | 34 | True |
| pick-out-of-hole-v3 | k30 | k20 | 6 | 5 | True |
| pick-place-v3 | k40 | k30 | 17 | 16 | True |
| pick-place-wall-v3 | pure | k50 | 26 | 26 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 50 | 48 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 48 | 50 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | koff | none | 31 | 33 | True |
| push-wall-v3 | k50 | k40 | 34 | 36 | True |
| reach-v3 | koff | none | 13 | 13 | True |
| reach-wall-v3 | k30 | k20 | 22 | 19 | True |
| shelf-place-v3 | pure | k50 | 8 | 8 | True |
| soccer-v3 | k30 | k20 | 12 | 10 | True |
| stick-pull-v3 | koff | none | 8 | 2 | True |
| stick-push-v3 | pure | k50 | 31 | 31 | True |
| sweep-into-v3 | koff | none | 16 | 6 | True |
| sweep-v3 | pure | k50 | 21 | 21 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 50 | 50 | True |
