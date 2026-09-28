# R5-D reproduction

Run from `/home/weiland/projects/openpi`. These programs read the cold store and
completed closed-loop logs; they do not start models, servers, or simulators.
All output goes beside the scripts. No output array exceeds 50 MB.

```bash
for script in diagnose_logs dynamics_probe contact_audit servo_probe noise_probe decision_audit aperture_floor contact_rule; do
  taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python "exp/offline_search/rounds/r05/ideation_D/${script}.py"
done
```

The programs run sequentially in one Python process at a time. `diagnose_logs`
reads all completed K7 summaries and the two `r4f_p_*_inf_s1001` controls.
It keeps accepted, error-free journal records with status `done` or `failed`,
matches attempts, and deduplicates `(uid, step)` across server logs. It saves
the first five executed actions on seven valid dimensions, and eight valid
state dimensions. Episode outcomes enter analysis only.

`dynamics_probe` fits each forward model using its own deployed library and
valid consecutive `next` edges. Five-fold validation holds out whole episodes
by episode index modulo five. `servo_probe` reads pre-existing R2 AWM results;
its pose estimate uses the logged top ten members, while its baseline action
is the saved exact synthesized segment. It is explicitly approximate and is
not a closed-loop replay.

`contact_audit` uses complete logged 16-member kernels on K7 anchor-tail blind
decisions. `contact_rule` independently reproduces its command-matched alarm
with online-legal arguments and measures CPU helper time. It is a pure helper,
not an installed Method or server change. `noise_probe` reads existing policy
resampling data; it runs no policy inference.

Principal machine-readable outputs:

- `inventory.json`, `logs_summary.json`: run coverage, accepted episode counts,
  task outcomes, stage-1 and MISS counts, owner-basis IR.
- `dynamics_library.json`, `feedback_logs.json`: model validation and feedback
  alarm incidence.
- `contact_audit.json`, `contact_episodes_*.json`, `contact_followups_*.json`:
  exact alarm definitions, task controls, timing, and subsequent actions.
- `contact_rule_validation.json`: independent alarm parity and CPU timings.
- `servo_probe.json`: pose correction direction/error checks.
- `noise_probe.json`: fixed-observation resampling and consecutive MISS counts.
- `decision_audit.json`: init-parity checks, action followups, measured file
  sizes, conservative representation additions, and sampling boundaries.
- `aperture_floor_library.json`: both models, both suites, both library sizes.

Input snapshot limitations: the hot store was absent, so the cold store
`/home/weiland/trace_runs/offline_search_store` was used. No completed
`r04_gblind` or `r04_rep` arm summary was present in the inspected inventory.
K10 had source and checks but no `HANDBACK.md` at inspection. The pure-policy
R4 controls lacked `robot_state`; no proprioceptive diagnosis is attributed
to those controls. The log inventory is a snapshot, not a claim about runs
which the coordinator completes subsequently.
