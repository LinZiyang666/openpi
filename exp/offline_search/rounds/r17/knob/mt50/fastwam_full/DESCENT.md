# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fast-WAM | 250 | 50/50 | 43.9% | 57.9% | 43.7% | 57.2% | 0.629 | False |
| Fast-WAM | 500 | 50/50 | 44.2% | 58.4% | 43.7% | 57.2% | 0.501 | False |
| Fast-WAM | 1000 | 50/50 | 43.4% | 57.9% | 43.7% | 57.2% | 0.505 | False |
| Fast-WAM | 1500 | 50/50 | 43.7% | 57.8% | 43.7% | 57.2% | 0.463 | False |
| Fast-WAM | 2000 | 50/50 | 43.3% | 57.9% | 43.7% | 57.2% | 0.440 | False |
| Fast-WAM | 2500 | 50/50 | 45.0% | 58.9% | 43.7% | 57.2% | 0.419 | False |

## Fast-WAM · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | pure | k50 | 5 | 5 | True |
| bin-picking-v3 | k50 | k40 | 16 | 14 | True |
| box-close-v3 | pure | k50 | 19 | 19 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 49 | 50 | True |
| button-press-v3 | k40 | k30 | 50 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 49 | 50 | True |
| coffee-button-v3 | k30 | k20 | 49 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 28 | 29 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 2 | 0 | True |
| disassemble-v3 | pure | k50 | 36 | 36 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 23 | 23 | True |
| door-open-v3 | koff | none | 43 | 36 | True |
| door-unlock-v3 | k40 | k30 | 44 | 43 | True |
| drawer-close-v3 | koff | none | 48 | 48 | True |
| drawer-open-v3 | k40 | k30 | 42 | 43 | True |
| faucet-close-v3 | pure | k50 | 44 | 44 | True |
| faucet-open-v3 | k30 | k20 | 49 | 50 | True |
| hammer-v3 | koff | none | 17 | 16 | True |
| hand-insert-v3 | pure | k50 | 12 | 12 | True |
| handle-press-side-v3 | koff | none | 45 | 44 | True |
| handle-press-v3 | pure | k50 | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 9 | 9 | True |
| handle-pull-v3 | pure | k50 | 19 | 19 | True |
| lever-pull-v3 | k40 | k30 | 27 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 13 | 13 | True |
| peg-unplug-side-v3 | k40 | k30 | 20 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 2 | 2 | True |
| pick-place-v3 | pure | k50 | 9 | 9 | True |
| pick-place-wall-v3 | pure | k50 | 21 | 21 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 40 | True |
| plate-slide-back-v3 | pure | k50 | 44 | 44 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | pure | k50 | 44 | 44 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | pure | k50 | 17 | 17 | True |
| push-wall-v3 | pure | k50 | 27 | 27 | True |
| reach-v3 | pure | k50 | 10 | 10 | True |
| reach-wall-v3 | pure | k50 | 34 | 34 | True |
| shelf-place-v3 | k40 | k30 | 6 | 8 | True |
| soccer-v3 | k30 | k20 | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 48 | 48 | True |
| sweep-into-v3 | pure | k50 | 20 | 20 | True |
| sweep-v3 | pure | k50 | 15 | 15 | True |
| window-close-v3 | k40 | k30 | 49 | 50 | True |
| window-open-v3 | k40 | k30 | 49 | 48 | True |
## Fast-WAM · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | k40 | k30 | 3 | 5 | True |
| bin-picking-v3 | k20 | k10 | 15 | 14 | True |
| box-close-v3 | k50 | k40 | 18 | 19 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 50 | 50 | True |
| button-press-v3 | k20 | k10 | 48 | 50 | True |
| button-press-wall-v3 | k20 | k10 | 50 | 50 | True |
| coffee-button-v3 | k30 | k20 | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 28 | 29 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 12 | 0 | True |
| disassemble-v3 | pure | k50 | 36 | 36 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 23 | 23 | True |
| door-open-v3 | koff | none | 45 | 36 | True |
| door-unlock-v3 | k30 | k20 | 43 | 43 | True |
| drawer-close-v3 | koff | none | 50 | 48 | True |
| drawer-open-v3 | k40 | k30 | 42 | 43 | True |
| faucet-close-v3 | k40 | k30 | 42 | 44 | True |
| faucet-open-v3 | k10 | koff | 48 | 50 | True |
| hammer-v3 | koff | none | 21 | 16 | True |
| hand-insert-v3 | pure | k50 | 12 | 12 | True |
| handle-press-side-v3 | k30 | k20 | 50 | 44 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 9 | 9 | True |
| handle-pull-v3 | pure | k50 | 19 | 19 | True |
| lever-pull-v3 | pure | k50 | 27 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 13 | 13 | True |
| peg-unplug-side-v3 | k30 | k20 | 19 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | pure | k50 | 9 | 9 | True |
| pick-place-wall-v3 | k50 | k40 | 23 | 21 | True |
| plate-slide-back-side-v3 | koff | none | 50 | 40 | True |
| plate-slide-back-v3 | k10 | koff | 50 | 44 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k40 | k30 | 42 | 44 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 17 | 17 | True |
| push-wall-v3 | pure | k50 | 27 | 27 | True |
| reach-v3 | k30 | k20 | 8 | 10 | True |
| reach-wall-v3 | k40 | k30 | 33 | 34 | True |
| shelf-place-v3 | k40 | k30 | 6 | 8 | True |
| soccer-v3 | koff | none | 3 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 48 | 48 | True |
| sweep-into-v3 | pure | k50 | 20 | 20 | True |
| sweep-v3 | pure | k50 | 15 | 15 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k30 | k20 | 47 | 48 | True |
## Fast-WAM · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | pure | k50 | 5 | 5 | True |
| bin-picking-v3 | k10 | koff | 14 | 14 | True |
| box-close-v3 | pure | k50 | 19 | 19 | True |
| button-press-topdown-v3 | k20 | k10 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k20 | k10 | 49 | 50 | True |
| button-press-v3 | k10 | koff | 49 | 50 | True |
| button-press-wall-v3 | k10 | koff | 49 | 50 | True |
| coffee-button-v3 | k20 | k10 | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 28 | 29 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 7 | 0 | True |
| disassemble-v3 | pure | k50 | 36 | 36 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 23 | 23 | True |
| door-open-v3 | k50 | k40 | 37 | 36 | True |
| door-unlock-v3 | k30 | k20 | 48 | 43 | True |
| drawer-close-v3 | koff | none | 50 | 48 | True |
| drawer-open-v3 | k30 | k20 | 46 | 43 | True |
| faucet-close-v3 | k40 | k30 | 45 | 44 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | koff | none | 19 | 16 | True |
| hand-insert-v3 | k50 | k40 | 10 | 12 | True |
| handle-press-side-v3 | k30 | k20 | 50 | 44 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 9 | 9 | True |
| handle-pull-v3 | pure | k50 | 19 | 19 | True |
| lever-pull-v3 | pure | k50 | 27 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 13 | 13 | True |
| peg-unplug-side-v3 | koff | none | 20 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 1 | 2 | True |
| pick-place-v3 | pure | k50 | 9 | 9 | True |
| pick-place-wall-v3 | k30 | k20 | 19 | 21 | True |
| plate-slide-back-side-v3 | k30 | k20 | 47 | 40 | True |
| plate-slide-back-v3 | k40 | k30 | 43 | 44 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k20 | k10 | 42 | 44 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 16 | 17 | True |
| push-wall-v3 | pure | k50 | 27 | 27 | True |
| reach-v3 | k50 | k40 | 10 | 10 | True |
| reach-wall-v3 | k30 | k20 | 34 | 34 | True |
| shelf-place-v3 | k30 | k20 | 6 | 8 | True |
| soccer-v3 | koff | none | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k50 | k40 | 46 | 48 | True |
| sweep-into-v3 | pure | k50 | 20 | 20 | True |
| sweep-v3 | pure | k50 | 15 | 15 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 48 | 48 | True |
## Fast-WAM · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | k30 | k20 | 3 | 5 | True |
| bin-picking-v3 | k20 | k10 | 16 | 14 | True |
| box-close-v3 | pure | k50 | 19 | 19 | True |
| button-press-topdown-v3 | koff | none | 49 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 49 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k10 | koff | 30 | 29 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 8 | 0 | True |
| disassemble-v3 | k40 | k30 | 36 | 36 | True |
| door-close-v3 | k20 | k10 | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 23 | 23 | True |
| door-open-v3 | koff | none | 40 | 36 | True |
| door-unlock-v3 | k30 | k20 | 44 | 43 | True |
| drawer-close-v3 | koff | none | 50 | 48 | True |
| drawer-open-v3 | koff | none | 45 | 43 | True |
| faucet-close-v3 | k50 | k40 | 43 | 44 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 16 | 16 | True |
| hand-insert-v3 | k50 | k40 | 12 | 12 | True |
| handle-press-side-v3 | koff | none | 42 | 44 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 9 | 9 | True |
| handle-pull-v3 | pure | k50 | 19 | 19 | True |
| lever-pull-v3 | k30 | k20 | 25 | 27 | True |
| peg-insert-side-v3 | k50 | k40 | 11 | 13 | True |
| peg-unplug-side-v3 | koff | none | 23 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | pure | k50 | 9 | 9 | True |
| pick-place-wall-v3 | k40 | k30 | 23 | 21 | True |
| plate-slide-back-side-v3 | k30 | k20 | 47 | 40 | True |
| plate-slide-back-v3 | k50 | k40 | 42 | 44 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k40 | k30 | 43 | 44 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k20 | k10 | 15 | 17 | True |
| push-wall-v3 | pure | k50 | 27 | 27 | True |
| reach-v3 | pure | k50 | 10 | 10 | True |
| reach-wall-v3 | k30 | k20 | 35 | 34 | True |
| shelf-place-v3 | k30 | k20 | 6 | 8 | True |
| soccer-v3 | koff | none | 4 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | pure | k50 | 48 | 48 | True |
| sweep-into-v3 | pure | k50 | 20 | 20 | True |
| sweep-v3 | pure | k50 | 15 | 15 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k10 | koff | 49 | 48 | True |
## Fast-WAM · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | k30 | k20 | 3 | 5 | True |
| bin-picking-v3 | pure | k50 | 14 | 14 | True |
| box-close-v3 | pure | k50 | 19 | 19 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k10 | koff | 28 | 29 | True |
| coffee-push-v3 | k50 | k40 | 19 | 21 | True |
| dial-turn-v3 | koff | none | 12 | 0 | True |
| disassemble-v3 | k40 | k30 | 35 | 36 | True |
| door-close-v3 | k20 | k10 | 48 | 50 | True |
| door-lock-v3 | pure | k50 | 23 | 23 | True |
| door-open-v3 | koff | none | 36 | 36 | True |
| door-unlock-v3 | k30 | k20 | 45 | 43 | True |
| drawer-close-v3 | koff | none | 50 | 48 | True |
| drawer-open-v3 | koff | none | 49 | 43 | True |
| faucet-close-v3 | k20 | k10 | 46 | 44 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | koff | none | 19 | 16 | True |
| hand-insert-v3 | k40 | k30 | 10 | 12 | True |
| handle-press-side-v3 | k10 | koff | 42 | 44 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 9 | 9 | True |
| handle-pull-v3 | pure | k50 | 19 | 19 | True |
| lever-pull-v3 | koff | none | 25 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 13 | 13 | True |
| peg-unplug-side-v3 | koff | none | 22 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 7 | 9 | True |
| pick-place-wall-v3 | k30 | k20 | 21 | 21 | True |
| plate-slide-back-side-v3 | k30 | k20 | 48 | 40 | True |
| plate-slide-back-v3 | pure | k50 | 44 | 44 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k40 | k30 | 44 | 44 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k40 | k30 | 17 | 17 | True |
| push-wall-v3 | pure | k50 | 27 | 27 | True |
| reach-v3 | k30 | k20 | 8 | 10 | True |
| reach-wall-v3 | k30 | k20 | 36 | 34 | True |
| shelf-place-v3 | k40 | k30 | 8 | 8 | True |
| soccer-v3 | k50 | k40 | 2 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k40 | k30 | 46 | 48 | True |
| sweep-into-v3 | pure | k50 | 20 | 20 | True |
| sweep-v3 | pure | k50 | 15 | 15 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 49 | 48 | True |
## Fast-WAM · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 19 | 19 | True |
| basketball-v3 | k30 | k20 | 4 | 5 | True |
| bin-picking-v3 | k40 | k30 | 13 | 14 | True |
| box-close-v3 | k50 | k40 | 21 | 19 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | k20 | k10 | 31 | 29 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 11 | 0 | True |
| disassemble-v3 | pure | k50 | 36 | 36 | True |
| door-close-v3 | k20 | k10 | 49 | 50 | True |
| door-lock-v3 | pure | k50 | 23 | 23 | True |
| door-open-v3 | k50 | k40 | 39 | 36 | True |
| door-unlock-v3 | k30 | k20 | 44 | 43 | True |
| drawer-close-v3 | koff | none | 50 | 48 | True |
| drawer-open-v3 | koff | none | 48 | 43 | True |
| faucet-close-v3 | k20 | k10 | 44 | 44 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | koff | none | 20 | 16 | True |
| hand-insert-v3 | k50 | k40 | 13 | 12 | True |
| handle-press-side-v3 | k20 | k10 | 45 | 44 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 9 | 9 | True |
| handle-pull-v3 | pure | k50 | 19 | 19 | True |
| lever-pull-v3 | koff | none | 28 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 13 | 13 | True |
| peg-unplug-side-v3 | koff | none | 21 | 18 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 7 | 9 | True |
| pick-place-wall-v3 | k40 | k30 | 24 | 21 | True |
| plate-slide-back-side-v3 | k30 | k20 | 47 | 40 | True |
| plate-slide-back-v3 | pure | k50 | 44 | 44 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k40 | k30 | 44 | 44 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 18 | 17 | True |
| push-wall-v3 | pure | k50 | 27 | 27 | True |
| reach-v3 | k30 | k20 | 8 | 10 | True |
| reach-wall-v3 | k30 | k20 | 36 | 34 | True |
| shelf-place-v3 | k40 | k30 | 12 | 8 | True |
| soccer-v3 | koff | none | 5 | 4 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k50 | k40 | 46 | 48 | True |
| sweep-into-v3 | pure | k50 | 20 | 20 | True |
| sweep-v3 | k30 | k20 | 15 | 15 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 48 | 48 | True |
