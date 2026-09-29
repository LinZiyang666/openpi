# Coordinator-only two-arm smoke recipe

This recipe has **not been executed**. Q2 ran CPU replays only. Use a fresh smoke run root and the coordinator's existing synchronization/server allocation procedure; do not reuse the pilot root or its active resources.

The supplied `emit_arms_smoke2.json` contains exactly:

| Arm | Method | Setting | Evaluation |
|---|---|---|---|
| `r6q2_pi05_l10_50_B_dose0p5` | `ExtraDosePi05` | deployed B + extra dose .5 | tasks 0,1 × original inits 0,1 = 4 episodes |
| `r6q2_groot_l10_50_risk_rho0p35` | `RiskLottery` | A + risk lottery, rho=.35, 10 controls | tasks 0,1 × original inits 0,1 = 4 episodes |

Both load the full model, disable native shadow, explicitly set `--os-root /home/weiland/trace_runs/offline_search_store`, and use `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only`. Client `replan_steps=5`; GR00T `resize_size=256`. No P3 or resampling flags are present. The same frozen randomization keys are used in smoke and the proposed full run, so changing the run directory or arm prefix does not change the coins.

All 43 deployment prefits already exist under `/tmp/q2_adapter_fits/`; `prefit_manifest.json` records their identities and SHA256s. To regenerate a missing prefit, run its command from `prefit_commands.json` or run `bash .../adapters/prefit_all43.sh`. These commands are CPU-only, run sequentially, and never start servers. Existing matching artifacts are verified and retained; conflicting artifacts cause an error.

For the coordinator, after replacing `<RUN>` below with a fresh smoke directory:

```bash
cd /home/weiland/projects/openpi
Q2_RUN='<RUN>'
Q2_ADAPTERS='exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters'
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python - "$Q2_ADAPTERS" "$Q2_RUN" <<'PY'
from pathlib import Path
import hashlib, json, shutil, sys
src, dst = map(Path, sys.argv[1:])
if '<RUN>' in str(dst):
    raise SystemExit('replace <RUN> with the coordinator-selected fresh smoke root')
dst.mkdir(parents=True, exist_ok=False)
(dst/'fits').mkdir()
(dst/'manifests').mkdir()
spec = json.loads((src/'emit_arms_smoke2.json').read_text().replace('<RUN>',str(dst.resolve())))
notes = {r['name']:r for r in json.loads((src/'prefit_manifest.json').read_text())}
for row in spec:
    note = notes[row['name']]
    fit = Path(note['artifact'])
    assert hashlib.sha256(fit.read_bytes()).hexdigest() == note['sha256']
    shutil.copy2(fit,dst/'fits'/fit.name)
    shutil.copy2(fit.with_suffix('.json'),dst/'fits'/fit.with_suffix('.json').name)
shutil.copy2(src/'smoke4_manifest.json',dst/'manifests/smoke4.json')
(dst/'arms_in.json').write_text(json.dumps(spec,indent=2)+'\n')
PY
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$Q2_RUN" --spec "$Q2_RUN/arms_in.json"
```

Then the coordinator uses the usual `ops/sync_remote.sh` and `ops/chain.sh` workflow for only the two named arms, with its assigned ports/CPUs/workers. The per-arm `manifest` makes `chain.sh` derive `EXPECT=4` and a manifest-specific DONE marker automatically. Clear any inherited `OSCL_MANIFEST` or Cartesian subset overrides; do not set a competing manifest. No port, host, worker, or chain was inspected or controlled by Q2.

Smoke acceptance (plumbing, not SR evidence):

1. Four accepted unique task/init pairs per arm; correct manifest-specific completion; no client/server error. Check both startup rows identify the new class, explicit store root, `shadow_native=false`, one policy-tail block, and the exact frozen kwargs/fit.
2. Every injected MISS logs `os_reason=61`, `os_q2_extra=1`, `os_q2_coin < os_q2_dose`. B-forced MISSes remain mandatory and log `os_q2_base_miss=1`, `os_q2_p_call=1`; extra dose applies only to `os_q2_eligible=1` anchors.
3. The GR00T first vision decision (`step=0`) logs `os_q2_episode_dose`, `os_q2_episode_propensity`, `os_q2_episode_coin`, `os_q2_task_p`, `os_q2_rho`, `os_q2_clamped`, and `os_q2_predicted_IR`; dose/propensity stay fixed through the episode. These fields are inside standard decision `extras`; join using `(uid, attempt, task_id, init)`. They are not additional top-level fields in the plugin's terminal `ev=episode` row.
4. Every ordinary non-final MISS is followed by exactly one zero-cost `policy_tail` request executing rows 5..9 of that policy chunk, then vision. Existing lifecycle protections can request vision early on malformed/nonfinite state or non-five-control execution; an injected call never bypasses them. Check GR00T gripper/action values are unchanged by the adapter. No source beyond row 9 is executed.
5. Reconcile ledger `N,V,M`: shadows/resamples are absent; blind/tail decisions contribute neither V nor M; owner IR is `.152*v+.848*m` or `.148*v+.852*m`. Do not demand that four episodes realize the nominal call fraction or certify a target SR.

For the full campaign, use `emit_arms_all43.json` and `eval500_manifest.json` instead, stage only their 43 fits, and point each manifest to `<RUN>/manifests/eval500.json`. Consult `../ADAPTERS.md` before finalizing the two 10-control lottery amendments and the redundant pi05 `A15` slot. Changing dose/rho/seed/kwargs requires a matching new prefit; the plugin correctly refuses a pickle whose kwargs differ. No full run is emitted or launched here.
