# Final handback — R15 track B

2026-10-05, astra. **Offline exploration and the full design are complete. Current result: NO-GO for online work.** Read [OFFLINE_GATES.md](OFFLINE_GATES.md) before implementation or pilot planning. The interim handback is preserved verbatim in [HANDBACK_INTERIM.md](HANDBACK_INTERIM.md).

A very light image query key is computationally feasible on CPU. The current library-only models do not retain enough accuracy on unseen recordings to replace the deployed key. No new-method closed-loop outcome was generated or read. Per the owner addendum, the pilot remains a plan and requires both an offline pass and coordinator approval.

## Main evidence

Primary candidate: symmetric action-aware spatial CNN, independently initialized and fully trained on CPU from each cell's own library, with both cameras and free state. No external pretrained weights or B-dev training.

| π0.5 L10 cell | Training rows / episodes | Full CPU fit | Library donor-LOEO RMS10 ratio | Unseen B-dev RMS10 excess | CPU key median / p95 |
|---|---|---:|---:|---:|---:|
| 50 | 2,925 / 50 | 80 epochs, 108.0 s | 0.9950 | +12.50% [11.13%, 13.93%] | 0.948 / 1.053 ms |
| 200 | 11,437 / 200 | 80 epochs, 372.2 s | 1.0168 | +10.00% [8.09%, 11.80%] | 0.971 / 1.044 ms |

RMS10 ratios/excess use equal episode weight within task and equal task weight, against the corresponding deployed-key geometry. Brackets are paired episode-bootstrap 95% intervals after development selection. B-dev contains 100 unseen episodes / 3,198 looks at 50 and 73 / 2,289 at 200. It consists of existing size-50 controller trajectories; size 200 is an offline library comparison, not a size-200 rollout. Donor-LOEO excludes the query episode from retrieval; the representation remains trained on the full cell library, so it is not unseen-encoder validation.

Both cells pass the numerical library-LOEO and local CPU latency screens, but fail the ≤5% B-dev excess, ≤10% upper-bound excess, every-task ≤15% excess and top-16 overlap gates. Size 50 additionally fails the teacher-phase discrepancy threshold. Local timing is one-thread TorchScript, both raw images through a usable CPU code, 400 timings per variant; all-task deployment export and H100-host timing remain pending. Eager CPU timing is about 1.19 ms. The full five-layer plugin has not been built or verified by this exploration.

The comparisons include symmetric spatial RGB PCA+AWM, asymmetric ridge distillation, symmetric/asymmetric CNN with and without action-pair supervision, a local residual kernel, direct action query keys, and state-only control. There are 20 reported variants/reference rows in each L10 cell and six in Spatial-50. Spatial has only 189 policy-labeled MISS looks; its 1,032 HIT looks lack policy labels, so its error result cannot clear the all-look gate. No explored eligible candidate clears the frozen B-dev gates.

## Economics and data limits

The cost equation is `IR_new = (epsilon * V + M) / N`, charging a complete policy including its tower on MISS. At epsilon = c_v/10 and fixed existing R14 counts, π0.5 L10-50 projects 30.9% saving at off, 17.1% at ir25, and 18.6% for the frozen task table. These are counterfactual savings, not measured new-method outcomes. The GR00T L10 task table projects only 0.69% saving at that epsilon; its high policy-call rate leaves little benefit even if accuracy were solved.

Exact π0.5 library RGB is available and sampled source alignment passed. GR00T build-library RGB is absent. The requested September trace directories are A-pool evaluation traces according to their launcher; only filenames and launcher source were inspected, never payloads. They are ineligible. Deterministic recovery requires the original simulator/reset/controller mapping and rowwise state plus object/image/token fidelity; it was assessed but not run. Size 500 has no unseen B complement.

All work after delivery of the CPU-preference addendum used CPU. Earlier four capped CNN GPU fits and two latency jobs completed before that interrupt under the original authorization, with a fresh ≥12-GiB-free check and 3-GiB allocator cap. CPU was feasible, so these are historical runs, not CPU-infeasibility exceptions. The primary models were subsequently trained afresh on CPU. Future GPU entry points require a recorded CPU-infeasibility reason.

## Deliverables and next responsibility

- [DATA.md](DATA.md): library/recording inventory, exact unseen selections, GR00T recovery assessment and source verification.
- [PROTOTYPE.md](PROTOTYPE.md), [metrics.csv](metrics.csv), [analysis.json](analysis.json): all candidates, donor-LOEO first, unseen B-dev errors, neighbors, phase proxies and uncertainty.
- [COST.md](COST.md), [cost_counts.csv](cost_counts.csv): CPU and historical GPU timings, epsilon bases, all R13/R14 uniform and per-task cost projections.
- [SPEC.md](SPEC.md): exact model/loss/training recipe, library-only layers 1–3 refits, feature-aware API, tower-free HIT, full-policy MISS and matched fixed caps.
- [GATES.md](GATES.md), [PREDICTION.md](PREDICTION.md): frozen accuracy/cost/SR thresholds, conditional economic pilot and explicit subjective forecasts.
- [REQUESTS.md](REQUESTS.md): sol's offline training/artifact/runtime/ledger implementation and verification contract; coordinator-only pilot conditions.
- [verification.json](verification.json), [final_verification.json](final_verification.json): source alignment, subset/sigma/teacher math, all saved donor exclusions, CPU checkpoint completeness, metric consistency and 52 reconciled B-arm cost ledgers. All 321 cost-source hashes rechecked unchanged.
- [FINAL_MANIFEST.json](FINAL_MANIFEST.json), [FINAL_MANIFEST.sha256](FINAL_MANIFEST.sha256): final content freeze of documents, scripts, model/evaluation artifacts and derived arrays, before any new-method closed-loop outcome. Runtime CUDA cache and the manifest's self-reference are excluded explicitly.

Sol's next useful milestone is a complete offline CPU artifact and plugin verification under this spec, accompanied by a revised model that clears the unchanged accuracy gates. Deployable implementation belongs to sol; rollout execution belongs to the coordinator. Nothing in this handback requests a run now. Open research questions are whether more spatial resolution/augmentation, temporal state or episode-cross-fitted student training can close the unseen gap using only the same library; these were not established here. Remaining data questions are exact GR00T frame recovery and all-look Spatial labels. No new online collection is authorized to answer them.

## Reproduction

Run from `/home/weiland/projects/openpi`. Every command uses this CPU-only prefix:

```bash
taskset -c 10-21,54-65 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Append the script and arguments below; script paths are relative to `exp/offline_search/rounds/r15/qk/astra/`. These are reproduction instructions, not commands to alter the frozen artifacts during review. Copy/version outputs in an authorized directory before any new experiment.

| Step | Script / arguments |
|---|---|
| Inventory and unseen selections | `inventory.py`; `recording_exclusions.py` |
| Own-cell train arrays | `prepare.py train --suite l10 --size 50` (repeat with 200; Spatial 50 also prepared) |
| Accepted B-dev arrays | `prepare.py dev --suite l10 --root r12_devlooks_50L`; Spatial uses `--suite spatial --root r12_devlooks_50S_local` |
| Linear/reference controls | `prototype.py --suite l10 --size 50` (repeat with 200) |
| Primary complete CPU fit + B-dev evaluation | `neural.py --suite l10 --size 50 --objective action --device cpu --epochs 80 --maxseconds 1800 --tag cnn_action_cpu` (repeat with 200) |
| Donor-LOEO comparison | `loeo.py --suite l10 --size 50` (repeat with 200; Spatial 50) |
| Primary CPU latency | `latency_cpu_extra.py --size 50` (repeat with 200; requires the saved direct-action control artifact) |
| Aggregate and apply gates | `analyze.py`; `gate_report.py`; `render.py` |
| Cost reconstruction | `cost_counts.py` |
| Focused consistency audit | `finalize.py` |

Prepared arrays are under `/home/weiland/trace_runs/offline_search_store/derived/r15_qk/`. All writes remained inside that root or this astra directory. No git, server, chain, remote operation, external-run wait, test-A root/queries read, or message to another agent was used. The manifest is the source of the final freeze timestamp and hashes.
