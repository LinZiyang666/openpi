# MT50 per-task knob descent

Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. A level passes with at most two fewer successes of 50. The first failure freezes the previous level; its measured episodes are the final result.

SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. A complete koff cell reaching the pure group-mean SR is used directly.

| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fast-WAM | 250 | 50/50 | 44.3% | 59.3% | 44.0% | 59.1% | 0.354 | False |
| Fast-WAM | 500 | 50/50 | 44.5% | 59.9% | 44.0% | 59.1% | 0.322 | False |
| Fast-WAM | 1000 | 50/50 | 43.5% | 59.1% | 44.0% | 59.1% | 0.266 | False |
| Fast-WAM | 1500 | 50/50 | 44.6% | 59.9% | 44.0% | 59.1% | 0.288 | False |
| Fast-WAM | 2000 | 50/50 | 44.6% | 59.8% | 44.0% | 59.1% | 0.268 | False |
| Fast-WAM | 2500 | 50/50 | 45.1% | 60.2% | 44.0% | 59.1% | 0.283 | False |

## Fast-WAM · size 250

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 16 | 16 | True |
| basketball-v3 | pure | k50 | 9 | 9 | True |
| bin-picking-v3 | k20 | k10 | 13 | 10 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | k30 | k20 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 49 | 50 | True |
| button-press-v3 | k20 | k10 | 49 | 50 | True |
| button-press-wall-v3 | k30 | k20 | 50 | 50 | True |
| coffee-button-v3 | k30 | k20 | 50 | 50 | True |
| coffee-pull-v3 | k10 | koff | 27 | 27 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 3 | 1 | True |
| disassemble-v3 | pure | k50 | 19 | 19 | True |
| door-close-v3 | k30 | k20 | 48 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | koff | none | 44 | 34 | True |
| door-unlock-v3 | pure | k50 | 47 | 47 | True |
| drawer-close-v3 | koff | none | 48 | 50 | True |
| drawer-open-v3 | k40 | k30 | 45 | 45 | True |
| faucet-close-v3 | pure | k50 | 48 | 48 | True |
| faucet-open-v3 | k30 | k20 | 49 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 18 | 18 | True |
| handle-press-side-v3 | koff | none | 45 | 42 | True |
| handle-press-v3 | pure | k50 | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 7 | 7 | True |
| handle-pull-v3 | pure | k50 | 27 | 27 | True |
| lever-pull-v3 | k50 | k40 | 25 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 12 | 12 | True |
| peg-unplug-side-v3 | pure | k50 | 25 | 25 | True |
| pick-out-of-hole-v3 | koff | none | 4 | 2 | True |
| pick-place-v3 | k30 | k20 | 6 | 8 | True |
| pick-place-wall-v3 | pure | k50 | 27 | 27 | True |
| plate-slide-back-side-v3 | pure | k50 | 50 | 50 | True |
| plate-slide-back-v3 | k40 | k30 | 34 | 35 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | pure | k50 | 46 | 46 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k40 | k30 | 19 | 20 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | k30 | k20 | 7 | 9 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | pure | k50 | 10 | 10 | True |
| soccer-v3 | k50 | k40 | 4 | 6 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 37 | 34 | True |
| sweep-into-v3 | pure | k50 | 23 | 23 | True |
| sweep-v3 | pure | k50 | 20 | 20 | True |
| window-close-v3 | k20 | k10 | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 48 | 48 | True |
## Fast-WAM · size 500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | pure | k50 | 16 | 16 | True |
| basketball-v3 | pure | k50 | 9 | 9 | True |
| bin-picking-v3 | k20 | k10 | 16 | 10 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | k20 | k10 | 48 | 50 | True |
| button-press-topdown-wall-v3 | k30 | k20 | 49 | 50 | True |
| button-press-v3 | k20 | k10 | 50 | 50 | True |
| button-press-wall-v3 | k20 | k10 | 50 | 50 | True |
| coffee-button-v3 | k30 | k20 | 50 | 50 | True |
| coffee-pull-v3 | k10 | koff | 28 | 27 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 12 | 1 | True |
| disassemble-v3 | koff | none | 18 | 19 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | koff | none | 44 | 34 | True |
| door-unlock-v3 | k40 | k30 | 45 | 47 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k20 | k10 | 44 | 45 | True |
| faucet-close-v3 | pure | k50 | 48 | 48 | True |
| faucet-open-v3 | k10 | koff | 48 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 18 | 18 | True |
| handle-press-side-v3 | k30 | k20 | 50 | 42 | True |
| handle-press-v3 | k30 | k20 | 49 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 7 | 7 | True |
| handle-pull-v3 | pure | k50 | 27 | 27 | True |
| lever-pull-v3 | pure | k50 | 27 | 27 | True |
| peg-insert-side-v3 | k40 | k30 | 11 | 12 | True |
| peg-unplug-side-v3 | pure | k50 | 25 | 25 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 6 | 8 | True |
| pick-place-wall-v3 | k40 | k30 | 28 | 27 | True |
| plate-slide-back-side-v3 | pure | k50 | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 33 | 35 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | pure | k50 | 46 | 46 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k40 | k30 | 22 | 20 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | k10 | koff | 7 | 9 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | pure | k50 | 10 | 10 | True |
| soccer-v3 | pure | k50 | 6 | 6 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 35 | 34 | True |
| sweep-into-v3 | pure | k50 | 23 | 23 | True |
| sweep-v3 | pure | k50 | 20 | 20 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k30 | k20 | 48 | 48 | True |
## Fast-WAM · size 1000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k20 | k10 | 15 | 16 | True |
| basketball-v3 | pure | k50 | 9 | 9 | True |
| bin-picking-v3 | k10 | koff | 15 | 10 | True |
| box-close-v3 | k50 | k40 | 24 | 24 | True |
| button-press-topdown-v3 | k20 | k10 | 50 | 50 | True |
| button-press-topdown-wall-v3 | k20 | k10 | 50 | 50 | True |
| button-press-v3 | k10 | koff | 49 | 50 | True |
| button-press-wall-v3 | k10 | koff | 49 | 50 | True |
| coffee-button-v3 | k20 | k10 | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 26 | 27 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 6 | 1 | True |
| disassemble-v3 | koff | none | 22 | 19 | True |
| door-close-v3 | koff | none | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | koff | none | 43 | 34 | True |
| door-unlock-v3 | k30 | k20 | 47 | 47 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | k30 | k20 | 46 | 45 | True |
| faucet-close-v3 | k50 | k40 | 48 | 48 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 18 | 18 | True |
| handle-press-side-v3 | k20 | k10 | 41 | 42 | True |
| handle-press-v3 | k30 | k20 | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 7 | 7 | True |
| handle-pull-v3 | pure | k50 | 27 | 27 | True |
| lever-pull-v3 | k40 | k30 | 25 | 27 | True |
| peg-insert-side-v3 | k50 | k40 | 10 | 12 | True |
| peg-unplug-side-v3 | pure | k50 | 25 | 25 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 8 | 8 | True |
| pick-place-wall-v3 | k40 | k30 | 26 | 27 | True |
| plate-slide-back-side-v3 | pure | k50 | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 35 | 35 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k40 | k30 | 45 | 46 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 18 | 20 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | k20 | k10 | 7 | 9 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | k50 | k40 | 9 | 10 | True |
| soccer-v3 | k50 | k40 | 4 | 6 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | koff | none | 32 | 34 | True |
| sweep-into-v3 | k50 | k40 | 21 | 23 | True |
| sweep-v3 | pure | k50 | 20 | 20 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k20 | k10 | 49 | 48 | True |
## Fast-WAM · size 1500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 18 | 16 | True |
| basketball-v3 | pure | k50 | 9 | 9 | True |
| bin-picking-v3 | k10 | koff | 9 | 10 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | koff | none | 49 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 49 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 27 | 27 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 7 | 1 | True |
| disassemble-v3 | koff | none | 25 | 19 | True |
| door-close-v3 | k20 | k10 | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | koff | none | 37 | 34 | True |
| door-unlock-v3 | k30 | k20 | 46 | 47 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 46 | 45 | True |
| faucet-close-v3 | pure | k50 | 48 | 48 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 18 | 18 | True |
| handle-press-side-v3 | koff | none | 42 | 42 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 7 | 7 | True |
| handle-pull-v3 | pure | k50 | 27 | 27 | True |
| lever-pull-v3 | k40 | k30 | 30 | 27 | True |
| peg-insert-side-v3 | k50 | k40 | 13 | 12 | True |
| peg-unplug-side-v3 | k40 | k30 | 28 | 25 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 8 | 8 | True |
| pick-place-wall-v3 | pure | k50 | 27 | 27 | True |
| plate-slide-back-side-v3 | pure | k50 | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 42 | 35 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | pure | k50 | 46 | 46 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k20 | k10 | 18 | 20 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | k20 | k10 | 7 | 9 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | k50 | k40 | 10 | 10 | True |
| soccer-v3 | pure | k50 | 6 | 6 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k50 | k40 | 32 | 34 | True |
| sweep-into-v3 | pure | k50 | 23 | 23 | True |
| sweep-v3 | pure | k50 | 20 | 20 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | k10 | koff | 49 | 48 | True |
## Fast-WAM · size 2000

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k30 | k20 | 16 | 16 | True |
| basketball-v3 | pure | k50 | 9 | 9 | True |
| bin-picking-v3 | koff | none | 17 | 10 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 27 | 27 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 13 | 1 | True |
| disassemble-v3 | koff | none | 28 | 19 | True |
| door-close-v3 | k20 | k10 | 48 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | k10 | koff | 35 | 34 | True |
| door-unlock-v3 | k30 | k20 | 47 | 47 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 45 | 45 | True |
| faucet-close-v3 | k50 | k40 | 47 | 48 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 18 | 18 | True |
| handle-press-side-v3 | koff | none | 41 | 42 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 7 | 7 | True |
| handle-pull-v3 | pure | k50 | 27 | 27 | True |
| lever-pull-v3 | koff | none | 26 | 27 | True |
| peg-insert-side-v3 | k50 | k40 | 14 | 12 | True |
| peg-unplug-side-v3 | pure | k50 | 25 | 25 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 8 | 8 | True |
| pick-place-wall-v3 | k40 | k30 | 26 | 27 | True |
| plate-slide-back-side-v3 | k40 | k30 | 48 | 50 | True |
| plate-slide-back-v3 | koff | none | 42 | 35 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | k40 | k30 | 45 | 46 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k40 | k30 | 19 | 20 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | k20 | k10 | 7 | 9 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | k50 | k40 | 8 | 10 | True |
| soccer-v3 | k50 | k40 | 4 | 6 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k40 | k30 | 32 | 34 | True |
| sweep-into-v3 | pure | k50 | 23 | 23 | True |
| sweep-v3 | pure | k50 | 20 | 20 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 48 | 48 | True |
## Fast-WAM · size 2500

| Task | Frozen level | First failure | Successes | Pure successes | Complete |
|---|---|---|---:|---:|---|
| assembly-v3 | k20 | k10 | 14 | 16 | True |
| basketball-v3 | pure | k50 | 9 | 9 | True |
| bin-picking-v3 | koff | none | 14 | 10 | True |
| box-close-v3 | pure | k50 | 24 | 24 | True |
| button-press-topdown-v3 | koff | none | 50 | 50 | True |
| button-press-topdown-wall-v3 | koff | none | 50 | 50 | True |
| button-press-v3 | koff | none | 50 | 50 | True |
| button-press-wall-v3 | koff | none | 50 | 50 | True |
| coffee-button-v3 | koff | none | 50 | 50 | True |
| coffee-pull-v3 | koff | none | 25 | 27 | True |
| coffee-push-v3 | pure | k50 | 21 | 21 | True |
| dial-turn-v3 | koff | none | 12 | 1 | True |
| disassemble-v3 | koff | none | 32 | 19 | True |
| door-close-v3 | k20 | k10 | 50 | 50 | True |
| door-lock-v3 | pure | k50 | 21 | 21 | True |
| door-open-v3 | koff | none | 35 | 34 | True |
| door-unlock-v3 | k30 | k20 | 45 | 47 | True |
| drawer-close-v3 | koff | none | 50 | 50 | True |
| drawer-open-v3 | koff | none | 47 | 45 | True |
| faucet-close-v3 | k50 | k40 | 46 | 48 | True |
| faucet-open-v3 | koff | none | 50 | 50 | True |
| hammer-v3 | pure | k50 | 31 | 31 | True |
| hand-insert-v3 | pure | k50 | 18 | 18 | True |
| handle-press-side-v3 | koff | none | 41 | 42 | True |
| handle-press-v3 | koff | none | 50 | 50 | True |
| handle-pull-side-v3 | pure | k50 | 7 | 7 | True |
| handle-pull-v3 | pure | k50 | 27 | 27 | True |
| lever-pull-v3 | koff | none | 26 | 27 | True |
| peg-insert-side-v3 | pure | k50 | 12 | 12 | True |
| peg-unplug-side-v3 | pure | k50 | 25 | 25 | True |
| pick-out-of-hole-v3 | koff | none | 0 | 2 | True |
| pick-place-v3 | k30 | k20 | 9 | 8 | True |
| pick-place-wall-v3 | k30 | k20 | 27 | 27 | True |
| plate-slide-back-side-v3 | pure | k50 | 50 | 50 | True |
| plate-slide-back-v3 | koff | none | 50 | 35 | True |
| plate-slide-side-v3 | koff | none | 50 | 50 | True |
| plate-slide-v3 | pure | k50 | 46 | 46 | True |
| push-back-v3 | koff | none | 0 | 0 | True |
| push-v3 | k30 | k20 | 18 | 20 | True |
| push-wall-v3 | pure | k50 | 36 | 36 | True |
| reach-v3 | k20 | k10 | 7 | 9 | True |
| reach-wall-v3 | pure | k50 | 42 | 42 | True |
| shelf-place-v3 | k40 | k30 | 8 | 10 | True |
| soccer-v3 | pure | k50 | 6 | 6 | True |
| stick-pull-v3 | koff | none | 0 | 0 | True |
| stick-push-v3 | k50 | k40 | 33 | 34 | True |
| sweep-into-v3 | pure | k50 | 23 | 23 | True |
| sweep-v3 | pure | k50 | 20 | 20 | True |
| window-close-v3 | koff | none | 50 | 50 | True |
| window-open-v3 | koff | none | 48 | 48 | True |
