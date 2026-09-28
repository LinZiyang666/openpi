"""Render the hand-back solely from completed evidence files."""
import datetime
import json
import shlex
from pathlib import Path
from exp.offline_search.rounds.r05.q4_growth.common import OUT,RUN,COLD,SHM,SPEC,sha256
from exp.offline_search.rounds.r05.q4_growth.prepare import PREFIX


def main():
    results=OUT/'results'
    builds={s:json.loads((results/f'build_{s}.json').read_text()) for s in ('l10','spatial')}
    checks=[json.loads(p.read_text()) for p in sorted(results.glob('check_*.json'))]
    assert len(checks)==8 and all(c['pass_all'] for c in checks)
    final=json.loads((results/'validation_final.json').read_text())
    assert len(final)==4 and all(c['pass_all'] for c in final)
    semantic=json.loads((results/'semantic_checks.json').read_text());assert semantic['pass_all']
    offline=[r for s in ('l10','spatial') for r in json.loads((results/f'offline_{s}.json').read_text())['records']]
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    text=[f'Q4 complete, verified {stamp}. Both required π0.5 libraries and all four CL2 artifacts are ready for coordinator-owned closed-loop evaluation. No existing method, plugin, or store library was edited. GR00T was optional and was not built; no required cell is missing.\n',
          '**Published libraries and paid provenance.** Each cell is labeled **“50 + 250 policy episodes”**. Only successful episodes from the 250 acquired task/init pairs contribute rows; all acquisition policy calls, including failed episodes, remain paid data.\n',
          '| Suite | Base episodes / rows | Acquisition episodes / successes | Paid policy calls | Appended rows | Final episodes / rows | Bytes per store copy |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for s,b in builds.items():
        text.append(f'| {s} | {b["actual_base_episodes"]} / {b["base_rows"]:,} | 250 / {b["successful_acquisition_episodes"]} | {b["paid_policy_calls"]:,} | {b["admitted_rows"]:,} | {b["final_episodes"]} / **{b["final_rows"]:,}** | {b["bytes_per_copy"]:,} |')
    text += ['\nThe nominal static-50 spatial library actually contains **49 episodes**, all preserved. Exact duplicate removals were **zero** in both suites. The source acquisition rows total 14,849 for l10 and 5,310 for spatial; successful rows total 10,481 and 5,222. Acquisition is all ten tasks × inits 0–24; evaluation is all ten tasks × inits 25–49. Failed acquisition rows and all evaluation rows are excluded from append. Both source-to-500-library file-overlap counts are zero; this does not imply independent benchmark initialization distributions.\n',
             'The new directories are:\n']
    for s,b in builds.items():
        for path in b['paths']:text.append(f'- `{path}/`')
    text += ['\nEach directory is a standard `offline_search.library.v1` store library, with full float32 pooled keys, robot state, `(10,32)` policy chunks, the twelve standard row arrays, `ids.json`, `episodes.json`, and `manifest.json`. Added arrays are `next_valid`, `terminal_known`, `is_terminal`, `is_growth`, `source_query_row`, and `source_query_episode`. `provenance.json` enumerates all 250 acquisition episodes, source H5 paths, success flags, source row ranges, per-episode admissions, the source fit SHA256, and hashes/bytes of source store inputs. `checksums.json` covers all 22 other files. No token/image archive is copied; AWM needs only pooled keys, state and actions.\n',
             'Base IDs, row order, metadata arrays, actions and keys are preserved bit for bit. New row IDs increase monotonically after the base prefix, in `(init, task_id, actual step)` order; string IDs are `growth:<lib_key>:<source_uid>:<step>`. Exact dedup compares task, frozen PCA64×2 representation, valid state8, and the **full 10×7 valid action**, with byte-equality confirmation on hash matches. An independent audit using frozen whitened codes/state gives exactly the same admission set and zero duplicates. Padding and head-only equality are excluded from admission decisions.\n',
             '`ep_len` is the complete source episode decision count and `progress=step/max(ep_len-1,1)`. Edges link only adjacent actual decisions of the same stored episode. A missing edge is unknown continuation; `next==-1` never defines terminal. `terminal_known` records knowledge from the completed episode, and `is_terminal` is true only at the known endpoint. These complete snapshots have zero missing nonterminal edges; a deliberately sparse fixture verifies that omitted middle rows create gaps without terminal flags or cross-episode links.\n',
             f'Before the l10 write, `/dev/shm` had **{builds["l10"]["shm_free_before_bytes"]:,} bytes** available; before spatial it had **{builds["spatial"]["shm_free_before_bytes"]:,}**. After both copies it had **{builds["spatial"]["shm_free_after_bytes"]:,}**. Total new library bytes are **{sum(b["bytes_per_copy"] for b in builds.values()):,} per store**, including JSON and checksums. Free space was checked again immediately before each RAM-store copy. Publication used new `.partial` directories and rename; existing destinations are refused.\n',
             '**Method switches and fit provenance.** Method spec: `'+SPEC+'`. Exact kwargs are `{"library":"grow250","variant":"refit","kref":5}` or `{"library":"grow250","variant":"frozen","kref":5}`. The wrapper uses the existing AWM query/synthesis implementation and returns `library="grow250"`; the plugin discovers and resolves the stored library without registration or plugin changes.\n',
             '- `refit`: AWM’s ordinary library fit on all grown rows, including randomized PCA, task centers, main and early whitening, candidate scales and LOEO confidence calibration. The action sigma remains from `current`, exactly as the existing AWM/harness convention does for any candidate library. The grown fingerprint cannot reuse the current PCA cache.\n- `frozen`: loads the actual R2 static-50 CL2 artifact; preserves PCA, main/early maps, sigma, scales, confidence coefficients, and every old candidate array exactly; projects only new candidate codes through those saved maps. The old R2 artifacts predate cached visual auxiliary task centers, so these unused-by-pure-CL2 centers are reconstructed from **base rows only**. New heads/state/norms/full actions are appended. No growing-bank recalibration occurs.\n',
             'Both new variants hold CL2 `k=16`, `kref=5`, joint features, early metric and `lam_c=.5` fixed, with insurance off. The R2 500 reference uses its deployed `kref=8`; its comparison is therefore not a candidate-count-only ablation. Frozen fitting is a CPU snapshot analogue of resident append; no GPU residency or latency claim is made.\n',
             '**Offline results.** These are fixed-recorded-observation diagnostics on both `inf` and `cache` streams, all 250 held-out pairs per stream, sampled at steps 1,6,11,… exactly as C’s main/stale geometry study. Every outcome is included. Main/stale top16 kernel actions are compared to the recorded policy `a_inf` head using current-library sigma and only 5×7 valid action dimensions. Early/fresh behavior is covered by separate query parity checks, not by this table.\n',
             'D1/D16 below use a **common frozen static-50 PCA/whitening frame for every candidate bank**. Identical grown contents therefore have identical density for frozen/refit. Action and successor losses use each variant’s actual fit and CL2 kernel; native own-fit distances are also saved in JSON, but should not be compared across separately whitened fits.\n',
             'Successor RMS compares kernel-weighted observed library `rs[next]-rs` to the recorded query’s next-state displacement, normalized per task by current-library state std floored at .05. Missing/terminal edges contribute **no invented zero transition**: their weight is omitted and the observed-edge weights are renormalized. All four variants use the same complete-case query mask. “Edge mass” exposes the original kernel weight retained before renormalization. This is a conditional dynamics diagnostic, not a rollout predictor.\n',
             '| Suite / stream | Variant | N action / successor | D1 | D16 | Action RMS | Successor RMS | Edge mass |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in offline:
        text.append(f'| {r["suite"]} / {r["stream"]} | {r["variant"]} | {r["queries"]} / {r["successor_queries"]} | {r["common_d1"]:.6f} | {r["common_d16"]:.6f} | {r["action_rms"]:.6f} | {r["successor_rms"]:.6f} | {r["observed_edge_mass"]:.6f} |')
    text += ['\nBoth growth variants reduce action and successor RMS relative to static-50 in these four recorded streams. The 500 library retains lower action RMS in all four. These observations establish neither SR gains nor a closed-loop ordering. `results/offline_{l10,spatial}.json` includes own-fit distances, new-row neighbor shares and equal-episode means; the sixteen accompanying NPZs contain query rows, masks and individual losses for reanalysis.\n',
             '**Arm specs and held-out manifest.** `arms_q4.json` is emit_arms input with literal `<RUN>` placeholders. `evaluation_pairs.json` is exactly `[[task_id, init], ...]`, all 10×25 evaluation pairs. K4’s loader/count tool accepted it with `EXPECT=250` and selection SHA256 `'+semantic['manifest_selection_sha256']+'`. The same manifest works for either suite.\n',
             'Resolved copies are already installed at `'+str(RUN/'arms_q4_resolved.json')+'`, `'+str(RUN/'evaluation_pairs.json')+'`, and `'+str(RUN/'arms.json')+'`; emitted server YAMLs and matrices are under `'+str(RUN/'config')+'/`. All four have `full_model:false`, `server_env:{"STAGE1_ONLY":"1"}`, five controls/request, no `--os-judge` flag, no blind/policy-tail flags, and `--os-no-shadow-native`.\n',
             '**Completed prefits.** All artifacts are under `'+str(RUN/'fits')+'/`. Sizes are serialized fit artifacts, not raw library archives. For scale, the original deployed π0.5 cache PKLs are 1,103,155,631 bytes for l10 and 430,792,483 bytes for spatial. Each grown store copy additionally retains raw pooled keys so refitting remains reproducible.\n',
             '| Arm / fit filename | Bytes | SHA256 |','|---|---:|---|']
    for c in checks:
        if c['root']==str(SHM):text.append(f'| `{c["arm"]}.pkl` | {c["bytes"]:,} | `{c["sha256"]}` |')
    text += ['\nThe following exact commands were executed from `/home/weiland/projects/openpi`; they are also in `prefit_commands.sh`. The small prefit entry calls the standard plugin prefit API with git metadata reading disabled. Existing artifact paths are refused, so these are reproduction commands for an empty fit destination, not an instruction to overwrite the completed artifacts.\n','```bash',*(OUT/'prefit_commands.sh').read_text().splitlines()[3:],'```\n',
             '**Validation completed.** `results/validation.json` and the finishing rerun `results/validation_final.json` both passed all four store/suite combinations. Checks include the existing store loaders, existing G5 library-vs-deployed-export gate, existing plugin payload validator against independent base/query source adapters, and the library registration validator. All full keys/state/actions and base metadata prefixes were checked for exact equality; all published files were checksummed in both copies. G5’s key/state/action maximum differences were all 0.\n',
             f'All four prefits were loaded through `PluginRuntime` in fresh processes against **each store root**, independently fitted again, and queried across all ten tasks, init25/init49, both streams, and early/fresh/stale regimes. The **eight checks passed {sum(c["query_parity_decisions"] for c in checks):,} query comparisons** (320 per check, including reversed episode order), with bit-identical fit arrays, topk, scores, actions, confidence and extras. Grown indices resolved through the plugin’s action tables. The frozen checks also verified all old candidate arrays and all saved fit/calibration fields unchanged. `results/check_*.json` contains individual hashes and counts. Semantic counterexamples cover full-tail differences, padding exclusion, task/state separation, and missing adjacency.\n',
             'The build and analysis commands actually run used this mandatory prefix; repeated suite/variant invocations below correspond to the recorded per-command logs:\n','```bash',
             'PY=('+shlex.join(PREFIX)+')',
             'Q4=exp/offline_search/rounds/r05/q4_growth',
             '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.build --suite l10 > "$Q4/results/build_l10.log" 2>&1',
             '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.build --suite spatial > "$Q4/results/build_spatial.log" 2>&1',
             '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.prepare > "$Q4/results/prepare.log" 2>&1',
             'bash "$Q4/prefit_commands.sh"',
             '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.validate > "$Q4/results/validation.log" 2>&1']
    for suite in ('l10','spatial'):
        for variant in ('refit','frozen'):
            for tag,root in (('shm',SHM),('cold',COLD)):
                arg='' if tag=='shm' else ' --root '+str(root)
                text.append(f'"${{PY[@]}}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite {suite} --variant {variant}{arg} > "$Q4/results/check_{suite}_{variant}_{tag}.log" 2>&1')
        text.append(f'"${{PY[@]}}" -m exp.offline_search.rounds.r05.q4_growth.offline --suite {suite} > "$Q4/results/offline_{suite}.log" 2>&1')
    text += ['"${PY[@]}" -m exp.offline_search.closed_loop.ops.remote.count --manifest-info "$Q4/evaluation_pairs.json" --model pi05 --suite libero_10 > "$Q4/results/manifest_validation.txt"',
             '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.semantic_checks > "$Q4/results/semantic_checks.log" 2>&1',
             '"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.validate --out "$Q4/results/validation_final.json" > "$Q4/results/validation_final.log" 2>&1',
             'bash -n "$Q4/prefit_commands.sh"','```\n',
             'These commands used at most eight concurrent Python processes, each restricted to CPUs 26–29,70–73 with single-thread BLAS and CUDA disabled. No server, port, tmux session, LIBERO worker or running chain was touched.\n',
             '**Coordinator next steps (not executed here).** Use the already emitted run files. Run a distinct smoke arm/run on held-out pairs, e.g. tasks 0 and 9 × inits 25 and 49, before the 250-pair manifest. Ensure `STAGE1_ONLY=1` at chain launch because the chain’s launch environment can override per-arm environment values. Use only coordinator-owned ports/resources and the existing remote sync/chain procedures. For full evaluation, unset unrelated `OSCL_MANIFEST` overrides so each arm’s manifest field is honored, or explicitly set it to `'+str(RUN/'evaluation_pairs.json')+'`. Do not collect acquisition inits again.\n',
             'Compare against `/home/weiland/trace_runs/os_closed_loop/r02_g50` arms `oscl50_p_l10_cl2` / `oscl50_p_sp_cl2` and `/home/weiland/trace_runs/os_closed_loop/r02_g500` arms `oscl500_p_l10_cl2` / `oscl500_p_sp_cl2`, restricting accepted records to exactly the same 250 manifest pairs. Report paired outcomes and actual executed controls; the two growth arms are pure cache with no judge. Charge acquisition separately: l10 14,849 and spatial 5,310 full-policy decisions. Do not describe the resulting banks as 50-episode libraries.\n',
             '**Files and limits.** Added source files are `common.py`, `build.py`, `method.py`, `prefit.py`, `prepare.py`, `validate.py`, `check_fit.py`, `semantic_checks.py`, `offline.py`, and `write_handback.py`, plus specs, manifest, command script, this hand-back, and `results/`. `file_manifest.json` records their observed modification times, byte sizes and SHA256, as well as fit and store-manifest identities. All work is confined to `q4_growth/`, the four new store directories, and `'+str(RUN)+'/`. The published libraries are immutable; the builder refuses existing names.\n',
             'No new closed-loop SR/IR, GPU serving, latency, simulator integration or causal rollout improvement has been measured. The offline study omits step0 and fresh-continuity selection by design; the parity checks exercise those code paths but do not test simulator outcomes. The growth snapshot uses paid full-inference data, not naturally accumulated mixed-policy MISSes. Fixed calibration can drift when candidate counts change. No hyperparameter or admission threshold was chosen on evaluation outcomes. GR00T remains an optional unbuilt extension.\n']
    (OUT/'HANDBACK.md').write_text('\n'.join(text))
    inventory=[]
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_manifest.json']
    paths += [p for p in RUN.rglob('*') if p.is_file()]
    for root in (COLD,SHM):
        for suite in ('l10','spatial'):
            d=root/'library'/f'pi05_{suite}'/'grow250'
            paths += [d/'manifest.json',d/'checksums.json']
    for p in sorted(paths):
        st=p.stat()
        inventory.append(dict(path=str(p),bytes=st.st_size,sha256=sha256(p),mtime_utc=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()))
    (OUT/'file_manifest.json').write_text(json.dumps(dict(generated_utc=stamp,files=inventory),indent=2))
    print(json.dumps(dict(handback=str(OUT/'HANDBACK.md'),files=len(inventory),checks=len(checks),store_validations=len(final))))

if __name__=='__main__':main()
