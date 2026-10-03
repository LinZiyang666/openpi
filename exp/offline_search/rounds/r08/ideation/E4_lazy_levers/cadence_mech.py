"""R8 callback E4: why do more frequent looks (A5, W5) and wrist-only looks (W10) lose on LIBERO-10?

Per arm (discovery inits 0-29 only), from live decisions and client controls:
  * look-to-look demo switching: top-weighted library episode changes between consecutive looks;
  * progress slip: |weighted library progress change - elapsed controls / top-demo length| between consecutive looks;
  * seam jumps: |delta action| (arm dims, robust per-dim scale of within-chunk deltas) at the first applied control of
    a decision whose chunk comes from a new look, vs inside chunks;
  * gripper command sign changes per 100 controls.
Writes /tmp/r8cb_E4_lazy_levers/cadence_mech.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader

RUN = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E4_lazy_levers')
LIB = {('pi05', 50): 'current', ('pi05', 500): 'bpool_cs', ('groot', 50): 'current', ('groot', 500): 'bpool_all'}


def analyze(arm, max_eps=300):
    a = reader.open_arm(RUN, arm)
    a.cache_enabled = False
    spec = a.manifest.get('arm_spec', {})
    model, suite = spec['model'], spec['suite_short']
    size = int(spec['r8'].get('library_size') or 50)
    cat = pd.read_parquet(RUN / 'catalog' / f"{model}_{'l10' if suite == 'l10' else 'spatial'}_{LIB[(model, size)]}" / 'rows.parquet').sort_values('row')
    ep_of, prog, eplen = cat.episode.to_numpy(), cat.progress.to_numpy(float), cat.ep_len.to_numpy(float)
    d = a.decisions(columns=['episode_key', 'decision_seq', 'init', 'vision', 'rows', 'weights', 'n_applied', 'src',
                             'anchor_decision_id', 'decision_id', 'journal_success', 'camera_mode'])
    d['init'] = d['init'].astype(int)
    d = d[d.init < 30]
    d['decision_seq'] = d.decision_seq.astype(int)
    d['vision'] = d.vision.astype(str).eq('True')
    d = d.sort_values(['episode_key', 'decision_seq'])
    switch, slip, gaps = [], [], []
    seam, inner, gflips, ctl_total, ys = [], [], [], 0, []
    keys = list(d.episode_key.unique())[:max_eps]
    for ek in keys:
        g = d[d.episode_key == ek]
        ys.append(str(g.journal_success.iloc[0]) == 'True')
        prev = None
        elapsed = 0
        newlook_seqs = set()
        for r in g.itertuples():
            n = int(float(r.n_applied)) if r.n_applied == r.n_applied and r.n_applied is not None else 5
            if r.vision and isinstance(r.rows, (list, np.ndarray)) and len(r.rows) == 16:
                rows = np.asarray(r.rows, int)
                w = np.asarray(r.weights, float)
                w = w / w.sum()
                top = np.bincount(np.searchsorted(np.unique(ep_of), ep_of[rows]), weights=w).argmax()
                p = float(w @ prog[rows])
                L = float(w @ eplen[rows])
                if prev is not None:
                    switch.append(top != prev[0])
                    slip.append(abs((p - prev[1]) - elapsed / max(5.0 * prev[2], 1.0)))
                    gaps.append(elapsed)
                prev = (top, p, L)
                elapsed = 0
                newlook_seqs.add(int(r.decision_seq))
            elapsed += n
        c = a.controls(ek, keys=['action', 'decision_seq', 'chunk_offset', 'is_settle'])
        act = np.asarray(c['action'], float)
        live = ~np.asarray(c['is_settle'], bool)
        act, seq, off = act[live], np.asarray(c['decision_seq'])[live], np.asarray(c['chunk_offset'])[live]
        if len(act) < 3:
            continue
        da = np.abs(np.diff(act[:, :6], axis=0))
        first = (seq[1:] != seq[:-1]) & np.isin(seq[1:], list(newlook_seqs))
        within = seq[1:] == seq[:-1]
        seam.append(da[first])
        inner.append(da[within])
        gs = np.sign(act[:, 6])
        gflips.append(int((gs[1:] != gs[:-1]).sum()))
        ctl_total += len(act)
    seam = np.concatenate(seam) if seam else np.zeros((0, 6))
    inner = np.concatenate(inner) if inner else np.zeros((0, 6))
    scale = np.median(inner, axis=0) + 1e-9
    return dict(arm=arm, episodes=len(keys), SR_discovery=float(np.mean(ys)),
                look_pairs=len(switch), median_controls_between_looks=float(np.median(gaps)) if gaps else None,
                demo_switch_rate=float(np.mean(switch)) if switch else None,
                progress_slip_p50=float(np.median(slip)) if slip else None,
                progress_slip_p90=float(np.quantile(slip, .9)) if slip else None,
                seam_jump_over_within_median=float(np.median((seam / scale).mean(1))) if len(seam) else None,
                seam_jump_mean=float((seam / scale).mean()) if len(seam) else None,
                within_jump_mean=float((inner / scale).mean()) if len(inner) else None,
                seams=int(len(seam)), gripper_sign_changes_per_100_controls=100.0 * sum(gflips) / max(ctl_total, 1))


def main():
    p = OUT / 'cadence_mech.json'
    out = json.loads(p.read_text()) if p.exists() else {}
    for arm in sys.argv[1:]:
        out[arm] = analyze(arm)
        p.write_text(json.dumps(out, indent=1))
        print(json.dumps(out[arm]), flush=True)


if __name__ == '__main__':
    main()
