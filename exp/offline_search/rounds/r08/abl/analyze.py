#!/usr/bin/env python3
"""r08_abl results: SR / owner IR per arm, paired exact McNemar against the SAME-TOPOLOGY controls (h100 servers),
plus the old A/B replicates (weilandserver + timan107) as a cross-topology reference only.

usage: python -m exp.offline_search.rounds.r08.abl.analyze [--out exp/offline_search/rounds/r08/abl/RESULTS.md]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path('/home/weiland/trace_runs/os_closed_loop')
RUNS = [ROOT / 'r08_abl', ROOT / 'r08_abl_t107']
PRICE = {'pi05': (.152, .848), 'groot': (.148, .852)}
CELLS = [(m, s, n) for m in ('p', 'g') for s in ('l10', 'sp') for n in (50, 500)]
GUARDS = ('stuck', 'terminal', 'overtime', 'no_progress')


def find(arm):
    for run in RUNS:
        d = run / 'runs' / arm
        done = (run / 'state' / f'{arm}.DONE').exists() or any((run / 'state').glob(f'{arm}.manifest_*.DONE'))
        if (d / 'summary.json').exists() and done:
            return run, d
    return None, None


def journal(d):
    out = {}
    for line in open(d / 'client' / 'journal.jsonl'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        uid = r.get('task_uid') or ''
        if ':eval:' not in uid or 'success' not in r or r.get('error') not in (None, ''):
            continue
        out[tuple(uid.split(':eval:')[1].split(':')[:2])] = bool(r['success'])
    return out


def owner_ir(d, model):
    led = json.load(open(d / 'summary.json')).get('cost_ledger') or {}
    if 'v' not in led or 'm' not in led:
        return None
    pv, pm = PRICE[model]
    return pv * led['v'] + pm * led['m']


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pair(ref, alt):
    keys = sorted(set(ref) & set(alt))
    b = sum(1 for k in keys if alt[k] and not ref[k])
    c = sum(1 for k in keys if ref[k] and not alt[k])
    delta = (sum(alt[k] for k in keys) - sum(ref[k] for k in keys)) / max(1, len(keys))
    return len(keys), b, c, delta, mcnemar(b, c)


def old_refs():
    """cell -> {'A': [run:arm]*3, 'B': [run:arm]*3} from the S-A / S-C provenance."""
    refs = {}
    for e in json.load(open(ROOT / 'r08_abl' / 'provenance.json')):
        m = 'p' if e['model'] == 'pi05' else 'g'
        s = 'sp' if e['suite'] in ('spatial', 'sp') else 'l10'
        refs.setdefault((m, s, e['scale']), {}).update({k: v for k, v in e['references'].items()})
    return refs


def load_spec(spec):
    run, arm = spec.split(':')
    return journal(ROOT / run / 'runs' / arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).with_name('RESULTS.md')))
    a = ap.parse_args()
    refs = old_refs()
    lines = ['# r08_abl results (standard mode, 500 official pairs per arm, h100 servers)', '',
             'Δ and p are paired exact McNemar against the **same-topology** control of the same cell (B control for',
             'guard arms, A control for CLIP; only-no-progress against both). "old A/B" = mean SR of the three R6',
             'replicates (weilandserver + timan107): cross-topology reference only. Owner IR = .152v+.848m (π0.5),',
             '.148v+.852m (GR00T).', '']
    rows_json = []
    for m, s, n in CELLS:
        model = 'pi05' if m == 'p' else 'groot'
        cell = f'{m}_{s}_{n}'
        arms = [f'r8abl_ctrlB_{cell}', f'r8abl_ctrlA_{cell}']
        if n == 500:
            arms += [f'r8abl_{g}_{cell}' for g in GUARDS]
        arms += [f'r8abl_onlynp_{cell}'] + ([f'r8abl_clip_{cell}'] if m == 'p' else [])
        data = {}
        for arm in arms:
            run, d = find(arm)
            if d is not None:
                data[arm] = (run, d, journal(d), owner_ir(d, model))
        ref_old = refs.get((m, s, n), {})
        old = {}
        for kind in ('A', 'B'):
            specs = ref_old.get(kind) or []
            srs = []
            for spec in specs:
                try:
                    j = load_spec(spec)
                    srs.append(sum(j.values()) / max(1, len(j)))
                except Exception:
                    pass
            old[kind] = sum(srs) / len(srs) if srs else None
        title = f"{'π0.5' if m == 'p' else 'GR00T'} {'LIBERO-10' if s == 'l10' else 'Spatial'}-{n}"
        oa = f"{old['A']:.3f}" if old['A'] is not None else '—'
        ob = f"{old['B']:.3f}" if old['B'] is not None else '—'
        lines += [f'## {title}  (old A {oa} / old B {ob})', '',
                  '| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |',
                  '|---|---|---|---|---|---|---|']
        cb = data.get(f'r8abl_ctrlB_{cell}')
        ca = data.get(f'r8abl_ctrlA_{cell}')
        for arm in arms:
            if arm not in data:
                lines.append(f'| {arm} | — | pending | | | | |')
                continue
            run, d, j, ir = data[arm]
            sr = sum(j.values()) / max(1, len(j))
            fleet = 'timan107' if run.name.endswith('t107') else 'timan108'
            cols = []
            row = dict(arm=arm, cell=cell, fleet=fleet, n=len(j), sr=sr, owner_ir=ir)
            for tag, ctrl in (('B', cb), ('A', ca)):
                if ctrl is None or ctrl[1] == d:
                    cols.append('')
                    continue
                k, b, c, delta, p = pair(ctrl[2], j)
                cols.append(f'{delta * 100:+.1f} pp (+{b}/−{c}, p={p:.3g})')
                row[f'vs_{tag}'] = dict(paired=k, plus=b, minus=c, delta=delta, p=p)
            irs = f'{ir:.3f}' if ir is not None else '—'
            lines.append(f'| {arm} | {fleet} | {len(j)} | {sr:.3f} | {irs} | {cols[0]} | {cols[1]} |')
            rows_json.append(row)
        lines.append('')
    Path(a.out).write_text('\n'.join(lines) + '\n')
    Path(a.out).with_suffix('.json').write_text(json.dumps(rows_json, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
