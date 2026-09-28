"""Append the demo extension, preserving the previous policy-growth hand-back."""
import datetime
import json
import shlex
from pathlib import Path
from exp.offline_search.rounds.r05.q4_growth.common import OUT,COLD,SHM,sha256
from exp.offline_search.rounds.r05.q4_growth.demo_prepare import RUN,SPEC
from exp.offline_search.rounds.r05.q4_growth.prepare import PREFIX

R=OUT/'results'/'demo'
MARK='**Demo-data scaling extension'

def main():
    builds={s:json.loads((R/f'build_{s}.json').read_text()) for s in ('l10','spatial')}
    validation=json.loads((R/'validation_final.json').read_text())
    assert len(validation)==4 and all(x['pass_all'] for g in validation for x in g['libraries'])
    semantics=json.loads((R/'semantics.json').read_text());assert all(x['pass_all'] for x in semantics)
    arms=json.loads((OUT/'arms_q4_demo.json').read_text())
    checks=[]
    for arm in arms:
        c=json.loads((R/f'check_{arm["name"]}.json').read_text())
        assert c['pass_all'] and sha256(c['artifact'])==c['sha256']
        checks.append(c)
    curves={s:json.loads((R/f'curve_{s}.json').read_text()) for s in ('l10','spatial')}
    assert all(len(x['records'])==20 for x in curves.values())
    table={(r['suite'],r['stream'],r['size'],r['variant']):r for x in curves.values() for r in x['records']}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    lines=[f'{MARK} — completed {now}.** All six requested demo libraries are published in both stores, both fit variants were evaluated at every size, and all eight requested anchor-tail prefits are ready. This section is separate from the paid policy-growth experiment above.\n',
           '**Corpus and nesting.** `bpool_cs` is the existing export of the August 2026 cache-OFF B-pool H5 collection (500 episodes per suite, 50 per task). It contains both successful and unsuccessful collection episodes: 436/500 successful l10 and 487/500 spatial. Selection retains both outcomes and uses no evaluation/query arrays or labels. The source metadata and source-file lists identify this as B-pool collection; the recorded query streams and the coordinator’s evaluation are A-pool. Numeric B-pool init IDs 0–49 are not A-pool evaluation trajectories.\n',
           'The deployed `current` episodes are **not contained** in `bpool_cs` in either suite: zero shared episode file paths, zero shared row IDs, and zero exact whole-episode key/state/action/step payload hashes. Current has 50 l10 episodes and 49 spatial episodes. Therefore **50 current → 100 demo changes collection as well as size**. Only **100 ⊂ 200 ⊂ 300 ⊂ 500** is a nested size series. `frozen50` transfers the deployed-50 representation onto the demo bank; it does not append demo data to current.\n',
           'Deterministic selection rule: within each task, sort the lowercase hexadecimal SHA256 of UTF-8 `q4_demo_nested_v1|<suite>|<task_id>|<stem>` ascending, with stem as the tie-break. Take the first 10/20/30 episodes for demo100/200/300. Concatenate complete episodes in `(rank, task_id, actual step)` order. IDs preserve the original `bpool_cs` strings; demo100 is an exact row/ID/array prefix of demo200, and demo200 of demo300. The existing 500 bank retains its original task-major row order; its membership is the superset and stable string IDs map shared rows. `provenance.json` records all selected episode files, source episode indices, source hashes and the exact rule. No success filter or deduplication changes these quotas.\n',
           '| Suite | Library | Episodes / per task | Successful episodes | Rows | Bytes per store copy |',
           '|---|---|---:|---:|---:|---:|']
    for s,b in builds.items():
        for r in b['libraries']:
            lines.append(f'| {s} | `{r["library"]}` | {r["episodes"]} / {r["episodes_per_task"]} | {r["success_episodes"]} | {r["rows"]:,} | {r["bytes_per_copy"]:,} |')
    lines += ['\nAll six names exist under **both** `/dev/shm/offline_search_store/library/pi05_<suite>/` and `/home/weiland/trace_runs/offline_search_store/library/pi05_<suite>/`. No `grow250`, `current`, `bpool_cs`, or other pre-existing store directory was changed. The new directories contain the standard full raw library arrays and metadata, plus source-row/episode maps and explicit `next_valid`, `terminal_known`, `is_terminal` flags. Complete source episodes preserve their real adjacency; missing edges are never inferred terminal. Tokens/images are omitted because these methods use pooled keys.\n',
              f'Total new demo-library storage is **{sum(sum(r["bytes_per_copy"] for r in b["libraries"]) for b in builds.values()):,} bytes per store**. `/dev/shm` free space was **{builds["l10"]["initial_shm_free_bytes"]:,} bytes** before l10 and **{builds["spatial"]["initial_shm_free_bytes"]:,}** before spatial; construction overlapped, and every individual copy rechecked free space. After all six it was **42,257,305,600 bytes**. Per-copy free-space measurements are in `results/demo/build_*.json`. Copies were published via new `.partial` directories and rename, with refusal on existing names.\n',
              'Reference bank row counts: current has **2,640 l10 / 1,018 spatial** rows; bpool_cs has **29,472 l10 / 10,909 spatial** rows. The nominal 50 spatial point retains the deployed 49-episode bank exactly.\n',
              '**Fit and controller rules.** Base method `exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoAWM` accepts `library` in `current,demo100,demo200,demo300,bpool_cs`, `variant` in `refit,frozen50`, and `kref=5`. **kref is fixed at 5 at every size and for both variants**, with k=16 and the other CL2 defaults unchanged. The 500 refit uses its own existing PCA basis and freshly recomputed AWM metric/calibration at kref=5; it is not the historical R2 kref=8 calibration. The 50 point is the same saved deployed CL2 fit for both labels.\n',
              '- `refit`: PCA, task centers, main/early whitening and confidence calibration use the selected candidate library, through the existing AWM fit recipe. As in AWM, action sigma is from current.\n- `frozen50`: PCA, main/early maps, task centers, sigma, s_c/s_d and confidence coefficients remain from current. Missing historical auxiliary task centers are reconstructed from current only. Candidate rows use a rowwise GEMV projection, independent of bank size, row position and row order. Shared candidate codes are **bit-identical across 100→200→300→500**, verified for every task in both suites. No candidate-size recalibration occurs.\n',
              'The arm method is `'+SPEC+'`: it composes the **unchanged K1 `_BlindMixin` used by `BlindAWM`** with `DemoAWM`. Exact controller kwargs are `serving="anchor_tail", budget=1, gates="budget_only"`. A vision anchor supplies the kernel chunk; one following blind decision executes its steps 5–9, then the budget requires vision again. No judge or policy tail is enabled. Frozen50 also preserves current’s K1 state scale and motion calibration; those are diagnostics under budget_only.\n',
              '**Offline curve.** All 500 A-pool episodes per stream, all ten tasks × inits 0–49, all outcomes; no subset manifest. Queries are steps 1,6,11,… as in C’s main/stale study. The action loss uses top16 kernel first-5×7 actions versus recorded a_inf in current sigma units. Successor loss uses weighted observed library state displacement versus the query’s observed next displacement, per-task current state std floored at .05; missing candidate edges are excluded and weights renormalized. All ten size/variant cells in a stream share one complete-case successor mask. The JSON/NPZ outputs include retained edge mass and equal-episode means.\n',
              'D1/D16 below use the **same frozen-50 geometry** for both fit labels at each size; they measure content density. They are computed from explicit Euclidean differences to avoid expanded-distance cancellation. Action/successor losses use each variant’s own fitted representation. Native distances in distinct whitening units are retained separately in JSON. An extra `tail_action_rms` compares an anchor’s predicted steps 5–9 with the next recorded policy head; this is also a fixed-observation diagnostic.\n',
              '| Suite / stream | Size | Queries / successor | Common D1 | Common D16 | Action refit | Action frozen50 | Successor refit | Successor frozen50 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in ('l10','spatial'):
        for stream in ('inf','cache'):
            for n in (50,100,200,300,500):
                a=table[s,stream,n,'refit'];b=table[s,stream,n,'frozen50']
                lines.append(f'| {s} / {stream} | {n} | {a["queries"]} / {a["successor_queries"]} | {a["common_d1"]:.6f} | {a["common_d16"]:.6f} | {a["action_rms"]:.6f} | {b["action_rms"]:.6f} | {a["successor_rms"]:.6f} | {b["successor_rms"]:.6f} |')
    lines += ['\nAcross the nested 100→500 series, mean action loss decreases at each size for both variants, both suites and both query streams. Refit action loss is lower than frozen50 at every demo size. Inference-query successor loss also decreases throughout; cache-query successor loss is not consistently monotone. In particular, the 500 frozen50 cache successor loss remains above current in both suites. These are fixed-observation measurements, not success-rate predictions.\n',
              'The complete measurements are `results/demo/curve_{l10,spatial}.json` and forty per-cell NPZs. Standalone figures are [loss curves](results/demo/demo_curve_losses.png) and [common-frame density curves](results/demo/demo_curve_density.png), also supplied as PDFs. The 50 point is deliberately disconnected from the nested demo series. Density monotonicity from 100 through 500 is checked per query; it does not require action or successor loss to improve monotonically. No offline metric is presented as SR or as a closed-loop ranking.\n',
              '**Arms and artifacts.** `arms_q4_demo.json` contains six refit arms (100/200/300 × l10/spatial) and two frozen50 arms (200 × l10/spatial). `<RUN>` placeholders resolve to `'+str(RUN)+'`. Already emitted: `arms_q4_demo_resolved.json`, `arms.json`, and `config/` in that run root. All eight are pure cache, `--os-blind`, no `--os-judge`, `full_model:false`, `STAGE1_ONLY=1`, and replan_steps=5. **There is no manifest field**: each arm targets all 500 evaluation pairs.\n',
              '| Prefit filename (under `'+str(RUN/'fits')+'/`) | Bytes | SHA256 |',
              '|---|---:|---|']
    for c in checks:lines.append(f'| `{Path(c["artifact"]).name}` | {c["bytes"]:,} | `{c["sha256"]}` |')
    lines += ['\nExact prefit commands were executed from the repo root and are stored in `demo_prefit_l10.sh` and `demo_prefit_spatial.sh`. Every command has the required affinity/thread/CUDA prefix. Existing artifacts are refused; do not blindly rerun these commands against populated fit paths. The final frozen50 commands were rerun after preserving earlier numerical-check artifacts under `r05_demo_curve/superseded/`; only the eight paths in `fits/` are advertised.\n','```bash']
    for s in ('l10','spatial'):lines += (OUT/f'demo_prefit_{s}.sh').read_text().splitlines()[3:]
    lines += ['```\n',
              '**Verification.** The initial and final store passes each validated all **12** suite/size/store copies: existing loaders, source-backed plugin payload validator, registration validator, current-library G5 gate, per-file hashes, complete source equality, IDs, progress and remapped adjacency. All six source subsets have zero missing nonterminal continuations. Both copies have identical checksum manifests.\n',
              f'The eight artifacts each passed a fresh-process existing **K1/plugin blind selftest** and its **independently fitted** logged replay: **{sum(c["plugin_selftest"]["decisions"] for c in checks)} decisions**, **{sum(c["plugin_selftest"]["vision"] for c in checks)} vision**, **{sum(c["plugin_selftest"]["blind"] for c in checks)} blind**, **zero MISSes**, and one broadcast per decision. Topk, scores, confidence, library, action, extras, look reason and dense history matched exactly. Each test uses two interleaved CPU orchestrators and four episode resets. Additional artifact load/query checks covered both stores, all ten tasks, both recorded streams and early/fresh/stale regimes: **{sum(sum(c["query_checks_by_root"].values()) for c in checks)} queries**. Frozen calibration, shared-code identity, exact tail-head equality and lifecycle reset/budget/MISS/task-change checks passed.\n',
              'A stricter intermediate test exposed float32 batching/row-order drift (2.15e-6 on an initial shared-code check, then 2.40e-5 at the reordered 500 bank). The final rowwise projection removes this drift rather than relaxing the shared-code equality check. Stale frozen artifacts temporarily failed replay against changed projection code; they were superseded, rebuilt and verified again. First-pass logs are preserved as diagnostics; the final `semantics.json`, `validation_final.json`, and `check_<arm>.json` are the acceptance evidence.\n',
              'Other exact commands run (the loops expand the individual suite/size invocations; all paths are under owned output directories):\n','```bash',
              'PY=('+shlex.join(PREFIX)+')','Q4=exp/offline_search/rounds/r05/q4_growth',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_build --suite l10 > "$Q4/results/demo/build_l10.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_build --suite spatial > "$Q4/results/demo/build_spatial.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_prepare > "$Q4/results/demo/prepare.log" 2>&1',
              'bash "$Q4/demo_prefit_l10.sh"','bash "$Q4/demo_prefit_spatial.sh"',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_validate > "$Q4/results/demo/validation.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_validate --out "$Q4/results/demo/validation_final.json" > "$Q4/results/demo/validation_final.log" 2>&1',
              'for suite in l10 spatial; do',
              '  "${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_offline --suite "$suite" > "$Q4/results/demo/offline_${suite}.log" 2>&1',
              '  for n in 100 200 300; do',
              '    "${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_check --suite "$suite" --size "$n" --variant refit --tag final > "$Q4/results/demo/check_${suite}_${n}_refit_final.log" 2>&1',
              '  done',
              'done',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_check --suite l10 --size 200 --variant frozen50 --tag final > "$Q4/results/demo/check_l10_200_frozen50_final.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_check --suite spatial --size 200 --variant frozen50 --tag rowwise > "$Q4/results/demo/check_spatial_200_frozen50_rowwise.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_semantics > "$Q4/results/demo/semantics.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_plot > "$Q4/results/demo/plot.log" 2>&1',
              '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_report',
              '```\n',
              '**Coordinator next steps and limits.** Use the emitted eight arms with the full 500-pair A-pool evaluation. Clear `OSCL_MANIFEST`, `OSCL_EPISODES` and `OSCL_TASKS` before a full run; do not reuse the previous policy-growth 250-pair manifest. Set `STAGE1_ONLY=1` in the launch environment, use coordinator-owned resources and the existing sync/chain procedure. Optional smoke recipe (not run): copy desired arms with `_smoke` names into a separate run root, retain the absolute active prefit paths, emit and sync; run with `OSCL_TASKS=0,9 OSCL_EPISODES=0 STAGE1_ONLY=1 WPS=1` and no manifest (two pairs per arm). Check initial vision, one blind decision per anchor, zero MISSes, and complete logs; keep smoke DONE markers separate from full-run arms. For a matched 500-library controller reference, hold kref=5; the historical K1 500 kref=8 arm has an additional kernel-setting difference.\n',
              'No servers, ports, tmux sessions, simulator workers, remote island, GPU or closed-loop chains were used. The plugin tests use fake policy observations and CPU orchestrators. No live SR/IR or serving-latency improvement is claimed. Single fixed nested selection gives one curve, not uncertainty over alternative demo subsets. All demo collection episodes are paid source data, not policy-growth acquisitions or free online learning. The 50→100 corpus change remains a confound. All requested cells were built.\n',
              '**Added files.** `demo_method.py`, `demo_build.py`, `demo_prepare.py`, `demo_validate.py`, `demo_offline.py`, `demo_check.py`, `demo_semantics.py`, `demo_plot.py`, `demo_report.py`, `arms_q4_demo.json`, the two `demo_prefit_*.sh` scripts, and `results/demo/`. `demo_file_manifest.json` records sizes, SHA256 and observed modification times for these, this extended hand-back, active run artifacts, and each new store manifest/checksum file. `results/demo/handback_original.md` preserves the prior hand-back exactly. No earlier method files or shared code were edited.\n']
    old=(OUT/'HANDBACK.md').read_text()
    prefix=old.split('\n\n'+MARK,1)[0]
    backup=R/'handback_original.md'
    if not backup.exists():backup.write_text(prefix)
    assert backup.read_text()==prefix
    full=prefix+'\n\n'+'\n'.join(lines)
    tmp=OUT/'HANDBACK.demo.tmp';tmp.write_text(full);tmp.replace(OUT/'HANDBACK.md')
    assert (OUT/'HANDBACK.md').read_text().startswith(backup.read_text())
    paths=[p for p in OUT.glob('demo*') if p.is_file() and p.name!='demo_file_manifest.json']
    paths += [OUT/'arms_q4_demo.json',OUT/'HANDBACK.md']
    paths += [p for p in R.rglob('*') if p.is_file() and '.mplconfig' not in p.parts]
    paths += [p for p in RUN.rglob('*') if p.is_file()]
    for root in (COLD,SHM):
        for s in ('l10','spatial'):
            for n in (100,200,300):
                d=root/'library'/f'pi05_{s}'/f'demo{n}';paths += [d/'manifest.json',d/'checksums.json']
    entries=[]
    for p in sorted(set(paths)):
        st=p.stat();entries.append(dict(path=str(p),bytes=st.st_size,sha256=sha256(p),mtime_utc=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()))
    (OUT/'demo_file_manifest.json').write_text(json.dumps(dict(generated_utc=now,previous_handback_sha256=sha256(backup),files=entries),indent=2))
    print(json.dumps(dict(handback=str(OUT/'HANDBACK.md'),libraries=6,store_copies=12,prefits=8,curve_cells=40,files=len(entries))))

if __name__=='__main__':main()
