"""Write the owner report and handback from verified preparation artifacts only."""
import json
from .boundary import HERE, NEW, install
from .offline import now
from exp.offline_search.rounds.r10.data import sha


def main():
    install()
    read = lambda name: json.loads((HERE / name).read_text())
    audit = read('final_audit.json'); assert audit['PASS']
    deploy = read('deployment.json')
    sources = read('source_files.json')
    tests = read('selftests.json'); assert len(tests) == 24 and all(t['PASS'] for t in tests)
    frozen = read('freeze.json')
    for p, digest in frozen['hashes'].items(): assert sha(p) == digest
    added = deploy['incremental_run_and_calibration_bytes']
    total = deploy['dependency_union_bytes']
    source_bytes = sum(s['bytes'] for s in sources['new_h100_sources'])
    summaries = read('offline_comparison.json')['cells']
    table = '\n'.join(f"| {s['cell_size']} | {s['loeo_change_pct']:+.2f}% | {s['dist_change_pct']:+.2f}% |"
                      for s in summaries)
    source_table = '\n'.join(f"| `{s['path']}` | {s['bytes']} | `{s['sha256']}` |" for s in sources['new_h100_sources'])
    receipt_times = {r.name: json.loads((r / 'emission_receipt.json').read_text())['timestamp_utc'] for r in NEW}
    report = f'''# 给 owner 的说明

已完成第三种修正器 GC_dist，共 24 个 arms，已经冻结并生成，尚未启动闭环评测。
全程没有读取任何旧闭环运行的客户端、流水、汇总、服务端日志或结果。只复用了允许的
R10 配置、缓存和修正头；训练、选规则和校准都来自 B 示范库。sol 的文件和已有 run root 未改动。

做法很简单：沿用 G 守卫和第二版 LOEO 修正头，离示范库近时最多修正一半，距离变远就逐渐收手。
距离达到校准标尺的两倍时不修正，第一步也不修正。每个模型、任务套件和库大小只增加一个距离标尺，
没有按具体任务另设参数，也没有增加训练样本。距离标尺通过整局留出、重新拟合训练局的距离度量得到，
没有拿含有查询局的度量来假装测量新的一局。

离线结果是小幅收益：24 组中 19 组比固定半强度的 LOEO 动作误差更低。各组等权平均，
LOEO 相对无修正降低 3.669%，GC_dist 降低 4.096%；GC_dist 相对 LOEO 平均降低约 0.424%。
GR00T 的 12 组全部改善。π0.5 最大两个库有取舍：500 局的长任务、空间任务分别比 LOEO 差约
0.85%、1.01%。这不是成功率，也不能承诺闭环会提高成功率。

规则是从预先列出的 18 个候选中，用同一批整局验证结果选出的。因此这些数字是调参验证结果，
不是额外独立测试。PCA 沿用当前 B 子库的固定基；每折的距离度量、动作尺度和修正头都重新拟合，
验证局不参与这些拟合。校准使用比最终部署更小的候选库，未做库大小外推，门限可能偏宽松。

预测在 {frozen['timestamp_utc']} 写下并加哈希封存，早于两个根目录的 emitter 输出。
24 个真实 CPU plugin 自检全部通过；两个标准 control plan 均已生成。没有进行同步、链式运行、
服务端、评测 worker 或远程启动。

## Frozen rule

For step > 0, `r = min geometric distance to retrieved rows / scale` and
`strength = 0.5 * clip((2.0-r)/1.25, 0, 1)`. Step 0 and nonfinite distances use zero.
Strength is .5 below r=.75, .4 at r=1, .2 at r=1.5, and zero at r>=2.
The same rule serves all 24 arms. `calibration/*.json` stores one scalar per cell-size.
These units are held-out-episode distance units, not Opus's in-library LOEO ratio.

The implementation subclasses sol's SizeController and RecipeCorrectedBase, preserves
the exact GC_loeo fit composition, and multiplies only the motion residual. The first
10 steps and channels 0–5 are corrected. Gripper, remaining steps, G behavior,
candidate retrieval, kernel, library indexing, and corrected-anchor continuation
follow the existing implementation. G can fire differently on later trajectories.
Exceptional regime-1 looks use geometric distances of the actual retrieved members.
One additional `_dist` calculation is made for each non-step-0 correction; no key
encoder, new store array, task gate, online adaptation, or additional policy call is added.

## Offline comparison

Metric: normalized 10-step six-motion MSE, averaged within each episode and then
over episodes. Five outer folds exclude whole episodes from head/metric fit and
donors. Four inner folds calibrate each outer training set's distance reference.
All 252,218 selected B rows are represented; nested sizes reuse the same episodes.
Failures are retained. Final deployment reuses sol's 240 original task heads exactly;
only CV heads are refitted. No PAIR or drift-augmented head is used.

| Cell-size | LOEO vs no corrector | GC_dist vs no corrector |
|---|---:|---:|
{table}

[OFFLINE.md](OFFLINE.md) includes absolute MSE, conditional episode-bootstrap intervals,
and the pooled distance bins. [distance_bins.csv](distance_bins.csv) contains every
cell-size breakdown. In calibrated r∈[2,3), LOEO is +0.86% vs no corrector on average;
at r≥3 it is +11.23% (19 cell-sizes have support). GC_dist is exactly zero change in
both bins. These are equal-cell-size means, with sparse far-tail support.
The offline no-corrector comparator is G's synthesis without the head, not a simulated
closed-loop guard trajectory. See [PROTOCOL.md](PROTOCOL.md) for selection details.

## Emission and validation

- pi05: `/home/weiland/trace_runs/os_closed_loop/r10_corr3_pi05` — 12 arms.
- GR00T: `/home/weiland/trace_runs/os_closed_loop/r10_corr3_groot` — 12 arms.
- Both manifests are byte-copied from corr2 and checked as tasks 0–9 × official inits 0–49.
- Standard emitter supplies all YAMLs and matrices. Full-model, policy-tail, guard-only,
  cost-ledger and client settings are copied from the corresponding corr2 GC_loeo arm.
- Frozen G/head equality audit: all 24 fits, every existing inner attribute/array unchanged;
  only the base class plus `distance_scale` and `distance_rule` are added.
- CPU integration: 24 passing selftests, 1,152 decisions, 48 connections, 96 episodes,
  96 forced no-progress triggers, 576 serving-correction checks.
- Additional gate audit: 576 peak/interior/zero/early/gripper/anchor checks; all pass.
- Both control plans pass; all 24 relocated fits pass path/head checks.
- Four semantic tests pass. Vectorized offline vs single-query serving maximum sampled
  chunk difference is 0.0001953; serving plugin checks use exact correction bytes.
- {audit['protected_inputs_unchanged']} protected source/spec/subset/G/head/GC files have unchanged hashes.

Synthetic CPU-test telemetry is under `astra/test_runs/`; no test telemetry is read
from an os_closed_loop run. The filesystem audit hook restricts old closed-loop reads
to allowed arms/manifest/fit/config files and all file writes to astra or the two new roots.
Its initial selftest attempt blocked dill's `/dev/null` type probe before inference;
allowing that character sink fixed the harness. No frozen serving rule or fit changed.

Emission receipts: `{json.dumps(receipt_times, sort_keys=True)}`.

## Deployment preparation

Extra h100 store bytes: **0**. New run/calibration destinations in the standard plans:
**{added:,} bytes ({added/(1<<30):.3f} GiB)**. The union of both complete dependency plans
is **{total:,} bytes ({total/(1<<30):.3f} GiB)**, including existing shared payload/store,
canonical G and head dependencies. The incremental count assumes those existing R10
dependencies are already available; no remote inventory was inspected.
The new serving sources add **{source_bytes:,} bytes**, outside the asset plans.
No transfer has occurred. Exact files, hashes and reasons are in each root's
`h100_sync/plan.json` and [deployment.json](deployment.json).

| New h100 source | Bytes | SHA256 |
|---|---:|---|
{source_table}

Only these two new Python files are needed at serving time, in addition to the already
required sol R10 and imported R9/R8/R4 runtime sources. Offline/build/audit helpers are
listed separately in [source_files.json](source_files.json). All new Python hashes are
in [ALL_SOURCES.sha256](ALL_SOURCES.sha256); runtime hashes in [H100_SOURCES.sha256](H100_SOURCES.sha256).

Prepared at {now()}. [PREDICTION.md](PREDICTION.md) and [freeze.json](freeze.json) remain unchanged.
'''
    (HERE / 'REPORT.md').write_text(report)
    handback = f'''# R10 astra handback

Status: complete, frozen blind to all closed-loop results; local CPU preparation only.
Start with [REPORT.md](REPORT.md), then [PREDICTION.md](PREDICTION.md).

## Frozen deliverable

24 GC_dist arms: 4 cells × 50/100/200/300/400/500 episodes, split 12 per root:

- `/home/weiland/trace_runs/os_closed_loop/r10_corr3_pi05`
- `/home/weiland/trace_runs/os_closed_loop/r10_corr3_groot`

Arm names: `r10_<model>_<suite>_<size>_GC_dist`.
Method: `exp.offline_search.rounds.r10.astra.method:DistanceController`.
Exact existing G and Stage 2b LOEO heads; distance attenuation only, no augmentation.
Strength `.5*clip((2-r)/1.25,0,1)`; step 0 disabled; one scalar calibration per cell-size.
All serving parameters and prediction frozen at **{frozen['timestamp_utc']}**.
`freeze.json` binds protocol, prediction, selection, comparison, 24 calibrations,
and the method/offline sources. Both `emission_receipt.json` timestamps are later.

Official A manifest is exactly 500 task/init pairs, byte-identical to corr2.
Existing R10 arms/fits/heads/source files were imported/read only, not edited.
No closed-loop client/journal/summary/server log or result was read, including R10.
No A/B-val recording was used. No sync, chain, server, worker, remote launch or git command ran.

## Evidence

- `PROTOCOL.md`: candidates and selection criterion stated before astra CV.
- `OFFLINE.md`, `offline_cv.csv`, `distance_bins.csv`, `offline_comparison.json`:
  full comparison, by-distance diagnostics and conditional uncertainty.
- `cv/*/folds.json`: all training/held-out episode identities and training-only scales.
- `cv/*/rows.npz`: row-level held-out sufficient statistics for every candidate.
- `selection.json`, `calibration/*.json`: single global selected gate, per-cell-size scalar.
- `fit_audit.json`: unchanged G/LOEO attributes and artifact SHA256 for all 24 arms.
- `selftests.json`: all 24 standard real-plugin CPU selftests pass, 1,152 decisions.
- `serving_gate_audit.json`, `relocation_audit.json`, `final_audit.json`: all pass.
- `input_hashes.json`: {audit['protected_inputs_unchanged']} protected inputs unchanged.
- `deployment.json`: extra store=0; new run/calibration={added:,} bytes;
  full dependency union={total:,} bytes. New serving sources={source_bytes:,} bytes.
- `H100_SOURCES.sha256`: the two new serving files; `ALL_SOURCES.sha256`: every helper.
- New-root `h100_sync/plan.json`: standard, complete bounded dependency plan.

Main offline result: GC_dist beats LOEO in 19/24 panels. Equal-panel MSE change vs
no correction is −4.096% vs LOEO −3.669%. All 12 GR00T panels improve; pi05 500
loses roughly 0.85%/1.01% relative to LOEO. This is tuning CV with a fixed PCA basis,
not an independent test and not a success-rate estimate. New-episode metric calibration
uses smaller donor folds without extrapolation; see the limitations in OFFLINE.md.

## Local checks only

All Python commands use `.venv/bin/python`, `PYTHONPATH=.:src`, CPUs `10-21,54-65`,
and OMP/OPENBLAS/MKL thread counts 1. Bytecode writes are disabled. The original
build was `astra.build build --workers 4`; new roots must not exist for that command.
Do not overwrite the frozen selection/calibrations/prediction in place.

Revalidate existing prepared artifacts, with no launch:

```sh
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.astra.tests
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.astra.build validate --workers 4
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.astra.audit
```

`run_tool.py plan <root> <12 explicit arm names>` invokes the unmodified standard
`closed_loop.ops.h100.control plan` under the file-access boundary. Both were run.
`run_tool.py selftest` imports sol's unmodified test; its synthetic logs stay in astra.
No remaining fitting, emission or local validation work is pending. Closed-loop
evaluation and any transfer remain the owner's separate task.
'''
    (HERE / 'HANDBACK.md').write_text(handback)
    print('REPORT and HANDBACK written')


if __name__ == '__main__':
    main()
