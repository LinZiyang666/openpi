# Experiment Artifact Layout (canonical)

All experiment-owned source code, configs, data, and analysis artifacts live
under `exp/<experiment>/` using a fixed four-slot sub-structure. This document
is the single source of truth for where new files go; contributors MUST match
existing files to these rules when adding to an experiment.

Introduced in the 2026-04-17 artifact-layout migration
(`logs/experiment_artifact_layout_plan.log.md`).

---

## 1. Canonical tree

Each experiment is self-contained. `exp/common/` is treated as an experiment
too — same sub-directory layout, but hosts cross-experiment shared assets.

```
exp/
  <experiment>/
    __init__.py
    <runner>.py, <helper>.py, ...   # experiment code directly at root (no code/ subdir)
    config/                         # all YAML configs for this experiment
    data/                           # run-produced artifacts: json, jsonl, log, h5
      cache_artifacts/              # pkl / DB / index artifacts
    analysis/                       # plot scripts + analysis tools + generated png/pdf/analysis.md
  common/                           # shared "experiment"
    <shared *.py>                   # e.g. _subprocess.py, _unit_key.py at root
    config/
    data/
      cache_artifacts/
    analysis/

tests/<exp>/test_*.py               # framework-integrated tests — live under tests/, not exp/
```

### 1.1 Experiment families (one optional nesting level)

When several independent experiments attack the same research question, they may
be grouped under a single **family** directory:

```
exp/
  <family>/
    README.md                       # family index: which experiments, what each concluded
    <experiment-a>/
      __init__.py
      <runner>.py, ...
      config/   data/   analysis/
    <experiment-b>/
      ...

tests/<family>/<experiment>/test_*.py
```

Rules:

- The family directory contains **only** `README.md` and experiment
  sub-directories. It must **not** hold code, config or data directly —
  otherwise the family/experiment boundary blurs over time and nobody can tell
  which artifacts belong to which experiment.
- Each experiment keeps the full four-slot structure of §1 unchanged.
- Exactly **one** nesting level is allowed; families of families are not.
- Test modules must have globally unique basenames (or their directory needs an
  `__init__.py`): `tests/` has no package markers and pytest's default `prepend`
  import mode makes two same-named test files collide at collection time.
- Introducing a family requires an entry in [`docs/README.md`](../README.md).

Current families: `exp/ablation_study/` (`executor_substitution`, `cache_size`, `latency_bench`).

### 1.2 Registry directories (not experiments)

A third kind of top-level `exp/` entry is permitted: a **registry** — a
directory that produces no experimental result and participates in no run, but
holds project-wide bookkeeping that must travel with the code.

```
exp/
  <registry>/
    README.md                       # how to query / add / verify
    <tool>.py                       # the code that reads and checks the ledger
    records/                        # the ledger itself, one file per subject
    analysis/                       # collected authoritative analysis output
      <task>/                       # one directory per research task — mandatory level
        MANIFEST.json               # every file: sha256 + the source it was taken from
        <figures, reports, plot data>
```

Rules:

- A registry **must not** carry `config/` or `data/`. Those two slots describe
  an experiment's inputs and its run products; a registry has neither, so
  shipping them empty would only invite misfiling.
- A registry **may** carry `analysis/`, and when it does the directory is
  organised as `analysis/<task>/` — **one sub-directory per research task, and
  that level is mandatory**. A flat `analysis/` turns into a pile of
  `pareto_combined.png` files whose owning experiment can only be guessed at;
  the task level is what keeps a collected figure attributable.
- Every `analysis/<task>/` **must** hold a `MANIFEST.json` naming each file with
  its `sha256` and the `source` path it was taken from. A collected figure with
  no recorded source is not authoritative — it is an orphan, and the whole point
  of a registry is that nothing in it is an orphan.
- Collection is a **copy, not a move**, unless the referring reports are updated
  in the same change: experiment `analysis/*.md` files link their figures by
  relative path, and moving the file silently breaks the published report.
- The ledger directory **must not** be named `data/`. §3 default-ignores every
  `exp/**/data/**` path, which would silently drop the ledger out of version
  control and void the registry's entire purpose. Use `records/` (or another
  non-`data` name).
- A registry stores **pointers and checksums, never bulk bytes**. The bytes stay
  on the node that produced them, named by the ledger.
- Design rationale lives in `logs/`, not here (Working Agreement §4). The
  registry's own `README.md` is an operations index only.
- Introducing a registry requires an entry in [`docs/README.md`](../README.md)
  and its own `logs/` design record.

Current registries: `exp/data_authority/` — which copy of an experiment dataset
is authoritative (node, path, sha256, content census, provenance, caveats).
Design record: [`logs/data_authority_plan.log.md`](../../logs/data_authority_plan.log.md).

## 2. File-kind rules

| Kind | Location |
|------|----------|
| Experiment runner / helper / builder `.py` | `exp/<exp>/` root |
| Experiment yaml config | `exp/<exp>/config/` |
| Run-time produced data (`*.json`, `*.jsonl`, `*.log`, `*.h5`, `*.csv`) | `exp/<exp>/data/` |
| Pkl / DB / index artifacts | `exp/<exp>/data/cache_artifacts/` |
| Plot scripts, analysis tools, `*.png`, `*.pdf`, analysis `*.md` | `exp/<exp>/analysis/` |
| Cross-experiment shared assets | `exp/common/` (same sub-structure) |
| Smoke / verification scripts integrated with pytest | `tests/<exp>/` (unchanged) |
| Smoke / verification scripts **not** integrated with pytest | alongside experiment code in `exp/<exp>/` root |

### Analysis vs data discrimination

When a directory mixes both kinds (a common pattern before the migration),
split by kind at move time:

- JSON / JSONL / CSV / per-run state → `data/`
- PNG / PDF / Markdown summary → `analysis/`
- Analysis scripts (`plot_*.py`, `analyze_*.py`) → `analysis/`

## 3. Tracking policy

`.gitignore` default-ignores every `exp/**/data/**` path to keep bulk
run-produced artifacts (pkl caches, HDF5, logs, Qdrant DBs) out of the repo.
The following paths are **explicit exceptions** whitelisted at the
`.gitignore` level and MUST stay tracked:

- `exp/trajectory_deviation/data/inits/*.{init,pruned_init}` — cross-machine
  source-of-truth for Step 1a initial states (shared across 1a/1b/2/3).
- `exp/trajectory_deviation/data/inits/per_unit_filters/*.json` — Step 1a
  unit filter definitions.
- `exp/common/data/cache_artifacts/**/calibration.json` — reproducible-run
  calibration metadata.
- The 24 experiment result JSONs listed in `.gitignore` under the
  "Tracked experiment result JSONs" block (phase1, trajectory,
  temporal_prune, warm_start `cache_eval_results.json` +
  `*.episode_results.json`, plus per-exp `experiment_state.json`).

If a new artifact needs tracking across machines, add it to `.gitignore`
as a `!exp/<exp>/data/...` line, not by hand-adding via `git add -f`.

## 4. When adding a new experiment

1. Create `exp/<new_exp>/__init__.py` with a 1-line docstring.
2. Add `config/`, `data/`, `analysis/` sub-dirs (with `__init__.py` only if
   Python imports them; leaf data/config dirs don't need it).
3. Put runners + helpers at `exp/<new_exp>/` root, not in a sub-package.
4. If the experiment reuses cache-build / run-experiment plumbing, import
   from `exp.common.*`, don't copy.

## 5. When moving a file between slots

- Tracked files: use `git mv` to preserve rename history.
- Untracked files: plain `mv` is fine.
- If splitting a directory by kind (data vs analysis), commit the split as
  one reviewable unit so `git log --follow` keeps working.

## 6. Out of scope

- Repo-root files (`cache.yaml`, `convert.py`, `cmd.sh`, etc.) stay at the
  root; they are not per-experiment.
- `assets/`, `src/`, `tests/`, `packages/` follow their own conventions.
- `scripts/` hosts framework-agnostic CLIs (`serve_policy.py`,
  `compute_norm_stats.py`, `train*.py`). Experiment-owned verify / dump
  scripts live in `exp/<exp>/` alongside their experiment code.

## 7. Verdict-factor enrichment in cache artifacts (B2)

`exp/common/build_in_memory_cache_artifact.py`,
`build_clip_cache_artifact.py`, and `build_llm_layer_matrix.py` accept an
optional `--factors-yaml` flag pointing at a minimal YAML (see
`docs/cache/verdict_factor_judge.md` §5) listing F1b OfflineWriter
factors. When set, the builder runs
`exp.common.factor_postprocess.enrich_artifact_with_factors` over the
finalized entry list, writes per-entry `payload.factors`, and stores the
artifact-level `LibraryStats` under the top-level `library_stats` key.
Without the flag, builders still compute `library_stats` from the entry
pool but write no factor descriptors — the resulting artifact is
backward-compatible with non-composite YAMLs.

Legacy artifacts that lack `library_stats` still load:
`InMemoryBackend.load_artifact` falls back to
`LibraryStats.compute_from_entries` at startup and logs a warning so
users notice and can rebuild with the new pipeline.
