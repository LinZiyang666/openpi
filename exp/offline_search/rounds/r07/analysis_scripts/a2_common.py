"""R7 A2 (mechanism / data quality) shared read-only helpers.

Sources: arm summary.json, client journal.jsonl + per_step.jsonl (client_timing rows), server
decisions_*.jsonl, runs/chain.log, C1's frozen stage tables (/tmp/r7_C1/stages/<cell>.pkl).
Nothing here writes to a run root. Run with the A2 prefix of ANALYSIS_BRIEF (CPUs 26-29,70-73).

Stage vocabulary (library-only, from the frozen C1 StageTable applied to a decision's retrieval kernel):
  macro  S0, S1, S2, S3, S4+ : weighted-majority `stage_run` of the known members (run index of the
                               executed-gripper two-mode segmentation; polarity-free: S0 = before the first
                               gripper change of the demonstration, S1 = after it, ...)
  cls    interior / event / mixed / unknown:
         unknown  = some member comes from a failed library episode (mode -1; SF/SW treat as hard)
         mixed    = members disagree on gripper mode (non-unanimous; a transition neighbourhood)
         event    = unanimous, but some member lies within one row of a gripper change (event_mass > 0)
         interior = unanimous and no member event-near
  label  "<macro>.<cls>"
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R7 = HERE.parent
ROOT = Path('/home/weiland/trace_runs/os_closed_loop')
EVAL = ROOT / 'r07_main'
PROFILE = ROOT / 'r07_profile_bval1'
STAGES = Path('/tmp/r7_C1/stages')
OUT = R7 / 'analysis_r7' / 'a2'          # compact machine-readable summaries
DETAIL = Path('/tmp/r7_A2')               # bulky per-anchor / per-episode tables
C1 = {'pi05': .152, 'groot': .148}
WRIST, COMPLETION = .055198, .049890      # R4 proportional-latency assumption (labelled)
LOOK_NAMES = {1: 'budget_cap', 6: 'lifecycle', 7: 'periodic', 8: 'c_ambiguous_or_judge', 11: 'follow_valve'}
HEAVY = ('served_head', 'robot_state', 'a_exec', 'topk', 'scores', 'winner')
CELLS8 = ['pi05_l10_50', 'pi05_l10_500', 'pi05_spatial_50', 'pi05_spatial_500',
          'groot_l10_50', 'groot_l10_500', 'groot_spatial_50', 'groot_spatial_500']


# ----------------------------------------------------------------------------- arms
def frozen_eval_arms():
    """SELECTION Freeze 1+2: 28 arms (SF1, UF1 x8; SW x4 pi0.5; CU30, CT30 x4 sparse)."""
    rows = []
    for cell in CELLS8:
        rows += [f'r7_{cell}_SF1', f'r7_{cell}_UF1']
        if cell.startswith('pi05_'):
            rows.append('r7_sw_pi05_' + cell[5:].replace('spatial_', 'sp_'))
        if cell.endswith('_50'):
            rows += [f'r7_{cell}_CU30', f'r7_{cell}_CT30']
    assert len(rows) == 28
    return rows


def parse_arm(name):
    """-> dict(model, suite l10|spatial, size 50|500, cell, r6cell, variant, family, profile)."""
    profile = name.endswith('_profile')
    core = name[:-len('_profile')] if profile else name
    m = re.fullmatch(r'r7_(sf_sw|sw)_pi05_(l10|sp)_(50|500)', core)
    if m:
        variant = 'SFSW' if m.group(1) == 'sf_sw' else 'SW'
        model, suite, size = 'pi05', 'l10' if m.group(2) == 'l10' else 'spatial', int(m.group(3))
    else:
        m = re.fullmatch(r'r7_(pi05|groot)_(l10|spatial)_(50|500)_([A-Za-z0-9]+)', core)
        if not m:
            raise ValueError(f'unrecognised arm name {name}')
        model, suite, size, variant = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    family = {'SF1': 'follow', 'SF2': 'follow', 'UF1': 'follow', 'SW': 'wrist', 'SFSW': 'wrist',
              'CU30': 'calls', 'CT30': 'calls', 'A': 'A'}.get(variant, 'other')
    cell = f'{model}_{suite}_{size}'
    return dict(arm=name, model=model, suite=suite, size=size, cell=cell, variant=variant, family=family,
                profile=profile, r6cell=f"{'l10' if suite == 'l10' else 'sp'}_{size}")


def done_markers(run_root, arm):
    """Manifest-bound markers (<arm>.manifest_<sha>.DONE) and plain ones (<arm>.DONE: chains launched without a
    selection manifest, e.g. r7_pi05_l10_50_SF1)."""
    st = Path(run_root) / 'state'
    return sorted(st.glob(f'{arm}.manifest_*.DONE')) + sorted(st.glob(f'{arm}.DONE'))


def arm_status(run_root, arm):
    run_root = Path(run_root)
    done = done_markers(run_root, arm)
    summ = (run_root / 'runs' / arm / 'summary.json').is_file()
    if done and summ:
        return 'complete'
    return 'running_or_missing' if (run_root / 'runs' / arm).exists() else 'missing'


def complete_arms(run_root, names):
    ok, skipped = [], {}
    for n in names:
        s = arm_status(run_root, n)
        (ok.append(n) if s == 'complete' else skipped.__setitem__(n, s))
    return ok, skipped


def profile_arms():
    return [r['arm'] for r in json.loads((PROFILE / 'arms.json').read_text())]


# ----------------------------------------------------------------------------- io
def jsonl(path, bad=None):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                if bad is not None:
                    bad[str(path)] += 1


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1, sort_keys=False) + '\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [clean(r) for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def pair_of(uid):
    t, i = uid.split(':eval:', 1)[1].split(':')[:2]
    return int(t), int(i)


# ----------------------------------------------------------------------------- loading
def load_arm(run_root, arm, keep_heavy=False):
    """Accepted episodes with their accepted-incarnation decision streams, plus QC counters.

    Accepted = journal row accepted & status done/failed & no error (collect.py rule). Decisions: same uid and
    attempt, ts <= journal ts, from the latest step-0 start (drops an earlier incarnation of the same attempt),
    exact duplicates collapsed; conflicts / gaps are recorded, never silently resolved.
    Also computes the ledger-style count (dedupe (uid, step) over the accepted attempt, last wins), which is what
    collect.py / cost_ledger count, so the two can be reconciled.
    """
    d = Path(run_root) / 'runs' / arm
    bad = Counter()
    jrows = list(jsonl(d / 'client/journal.jsonl', bad))
    acc, qc = {}, Counter()
    for j in jrows:
        qc['journal_rows'] += 1
        if j.get('error'):
            qc['journal_error_rows'] += 1
        if j.get('accepted') and j.get('status') in ('done', 'failed') and not j.get('error'):
            u = j['task_uid']
            if u in acc and acc[u] != j:
                qc['journal_conflicting_accepted'] += 1
            acc[u] = j
    groups, ledger_style = defaultdict(list), {}
    starts = []
    for f in sorted(glob.glob(str(d / 'server_*' / 'decisions_*.jsonl'))):
        for r in jsonl(f, bad):
            ev = r.get('ev')
            if ev == 'startup':
                starts.append({k: r.get(k) for k in ('tag', 'method', 'model', 'stage1_mode', 'request_cameras',
                                                     'camera_cost_basis', 'kwargs', 'git_head')})
                continue
            if ev != 'dec':
                continue
            qc['decisions_logged_all'] += 1
            j = acc.get(r.get('uid'))
            if not j:
                qc['decisions_nonaccepted_uid'] += 1
                continue
            if int(r.get('attempt', 1) or 1) != int(j.get('attempt', 1) or 1):
                qc['decisions_other_attempt'] += 1
                continue
            if not keep_heavy:
                for k in HEAVY:
                    r.pop(k, None)
            ledger_style[(r['uid'], r['step'])] = r
            if r.get('ts', 0) > j.get('ts', math.inf):
                qc['decisions_after_journal_ts'] += 1
                continue
            groups[r['uid']].append(r)
    episodes = []
    for uid, j in sorted(acc.items()):
        ds = groups.get(uid, [])
        s0 = [x.get('ts', 0) for x in ds if x['step'] == 0]
        if not s0:
            qc['episodes_missing_stream'] += 1
            continue
        start = max(s0)
        kept = [x for x in ds if x.get('ts', 0) >= start]
        qc['decisions_prior_incarnation'] += len(ds) - len(kept)
        uniq = {}
        for x in kept:
            s = int(x['step'])
            if s in uniq:
                a, b = uniq[s], x
                same = all(a.get(k) == b.get(k) for k in ('vision', 'hit', 'src', 'rows', 'weights', 'look_reason'))
                qc['duplicate_identical' if same else 'duplicate_conflicting'] += 1
                if not same:
                    continue
            uniq[s] = x
        steps = sorted(uniq)
        if steps != list(range(len(steps))):
            qc['episodes_gapped'] += 1
        t, i = pair_of(uid)
        episodes.append(dict(uid=uid, task=t, init=i, Y=int(bool(j['success'])), attempt=int(j.get('attempt', 1) or 1),
                             run_id=j.get('run_id'), journal_ts=j.get('ts'), decisions=[uniq[s] for s in steps]))
    lv = list(ledger_style.values())
    ledger_counts = dict(N=len(lv), V=sum(bool(r.get('vision')) for r in lv), M=sum(r.get('hit') is False for r in lv))
    qc['malformed_lines'] = sum(bad.values())
    try:
        info = parse_arm(arm)
    except ValueError:
        info = None  # reference arms of earlier rounds
    return dict(arm=arm, run_root=str(run_root), info=info, accepted=acc, episodes=episodes,
                qc=dict(qc), ledger_style=ledger_counts, startups=starts)


def client_timing(run_root, arm):
    """(task_uid, run_id, attempt) -> list of client_timing rows (termination_reason, infers, steps, success)."""
    p = Path(run_root) / 'runs' / arm / 'client' / 'per_step.jsonl'
    out, steps = defaultdict(list), Counter()
    if not p.exists():
        return out, steps, False
    with open(p) as f:
        for line in f:
            if '"_kind"' in line and '"client_timing"' in line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('_kind') == 'client_timing':
                    out[(r.get('task_uid'), r.get('run_id'), int(r.get('attempt', 1) or 1))].append(r)
            elif '"step_idx"' in line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('_kind') is None and 'step_idx' in r:
                    steps[(r.get('task_uid'), r.get('run_id'), int(r.get('attempt', 1) or 1), r.get('hit_type'))] += 1
    return out, steps, True


def chain_events(run_root):
    """Parse runs/chain.log 'EV <ts> <EVENT> arm=<arm> ...' lines -> {arm: Counter(event)} and EXC_PURGED n."""
    run_root = Path(run_root)
    ev, purged = defaultdict(Counter), defaultdict(int)
    lines = []
    for p in [run_root / 'runs' / 'chain.log', *sorted(run_root.glob('chain_console_*.log'))]:
        if p.exists():
            lines += [x for x in p.read_text(errors='replace').splitlines() if x.startswith('EV ')]
    for line in dict.fromkeys(lines):  # console logs repeat chain.log lines; count each event once
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[2]
        kv = dict(x.split('=', 1) for x in parts[3:] if '=' in x)
        arm = kv.get('arm') or (parts[3] if name == 'CHAIN_DONE' and len(parts) > 3 else None)
        ev[arm][name] += 1
        if name == 'EXC_PURGED':
            purged[arm] += int(kv.get('n', 0))
    return ev, purged


# ----------------------------------------------------------------------------- stages
@lru_cache(None)
def stage_table(cell):
    from exp.offline_search.rounds.r07.stages.stages import StageTable
    return StageTable.load(STAGES / f'{cell}.pkl')


def macro_name(run):
    return 'S?' if run is None or run < 0 else ('S4+' if run >= 4 else f'S{run}')


def stage_of(table, rows, weights):
    rows = np.asarray(rows, np.int64)
    w = np.asarray(weights, float)
    if not len(rows) or rows.max() >= len(table.mode) or rows.min() < 0:
        return dict(cls='invalid', macro='S?', label='S?.invalid')
    o = table.online(rows, w)
    modes = table.mode[rows]
    known = modes >= 0
    wn = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    run = None
    if known.any():
        runs = table.stage_run[rows[known]]
        mass = Counter()
        for r_, ww in zip(runs.tolist(), wn[known].tolist()):
            mass[int(r_)] += ww
        run = max(sorted(mass), key=lambda k: mass[k])  # ties -> lowest run
    if o['unknown_mass'] > 0:
        cls = 'unknown'
    elif not o['unanimous']:
        cls = 'mixed'
    elif o['event_mass'] > 0:
        cls = 'event'
    else:
        cls = 'interior'
    macro = macro_name(run)
    return dict(cls=cls, macro=macro, label=f'{macro}.{cls}', run=run, unanimous=bool(o['unanimous']),
                event_mass=float(o['event_mass']), unknown_mass=float(o['unknown_mass']),
                mode0_mass=float(o['mode0_mass']), min_rows_to_event=int(o['min_rows_to_event']))


def ex(d, key):
    e = d.get('extras') or {}
    return e.get(key)


def bx(d, key):
    e = d.get('blind_extras') or {}
    return e.get(key)


# ----------------------------------------------------------------------------- references
def a_reference_specs(info, profile_ref=False):
    """A three replicates on the same 500 pairs (R6 identities) or, for PROFILE, the paired single A_profile arm."""
    if profile_ref or info['profile']:
        return [f"{PROFILE.name}:r7_{info['cell']}_A_profile"]
    from exp.offline_search.rounds.r06.analysis_scripts import common as R6
    a, _ = R6.ab_arms(info['model'], info['r6cell'])
    return a


@lru_cache(None)
def journal_outcomes(spec):
    """{(task, init): 0/1} from accepted terminal rows of run:arm (collect.py rule)."""
    run, arm = spec.split(':')
    out = {}
    for j in jsonl(ROOT / run / 'runs' / arm / 'client/journal.jsonl'):
        if j.get('accepted') and j.get('status') in ('done', 'failed') and not j.get('error'):
            k = pair_of(j['task_uid'])
            v = int(bool(j['success']))
            if k in out and out[k] != v:
                raise ValueError(f'conflicting accepted outcomes {spec} {k}')
            out[k] = v
    return out


def wilson(k, n, z=1.959964):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def cli_arms(args):
    """Resolve --run/--arms into (run_root, [complete arms], {skipped: status})."""
    run_root = EVAL if args.run == 'eval' else PROFILE if args.run == 'profile' else Path(args.run)
    if args.arms:
        names = args.arms
    elif run_root == EVAL:
        names = frozen_eval_arms()
    else:
        names = profile_arms()
    if getattr(args, 'family', None):
        names = [n for n in names if parse_arm(n)['family'] in args.family]
    ok, skipped = complete_arms(run_root, names)
    return run_root, ok, skipped
