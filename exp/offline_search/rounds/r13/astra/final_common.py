"""CPU-only reporting helpers; no controller fitting or rollout imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
COSTS = {'groot': (.148, .852), 'pi05': (.152, .848)}
SEED, BOOT = 20261004, 5000
INPUTS = {}
FROZEN = {line.split(maxsplit=1)[1].lstrip('*'): line.split()[0]
          for line in (HERE / 'MANIFEST.sha256').read_text().splitlines()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_frozen():
    for name, digest in FROZEN.items():
        assert sha(HERE / name) == digest, name
    return {'members': len(FROZEN), 'all_match_manifest': True}


def record(path):
    path = Path(path)
    INPUTS[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read(path):
    return json.loads(record(path).read_text())


def jsonlines(path):
    with record(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(name, value):
    path = (HERE / name).resolve()
    rel = str(path.relative_to(HERE))
    assert rel not in FROZEN, f'Frozen output prohibited: {rel}'
    assert rel in ('ANALYSIS.md', 'REPORT.md') or rel.startswith(('final_', 'final/'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def dump(name, obj):
    write(name, json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv(name, rows):
    write(name, pd.DataFrame(rows).to_csv(index=False))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def cell_label(cell):
    return cell.replace('groot_', 'GR00T ').replace('pi05_', 'π0.5 ').replace('l10_', 'L10-').replace('spatial_', 'Spatial-')


def pct(x):
    return f'{100*x:.1f}%'


def pp_ci(r):
    return f"{100*r['difference']:+.2f} [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]"


def paired_stats(rows):
    """Pairs are weighted equally; original init clusters carried across cells/doses/reps.

    Stratified paired cluster percentile bootstrap, conditional on benchmark tasks
    and observed replicate blocks. Different B subsets can give variable denominators.
    """
    a = pd.DataFrame(rows)
    a['diff'] = a.success_d - a.success_p
    groups = a.groupby(['suite', 'task', 'init'], sort=True).agg(
        numerator=('diff', 'sum'), denominator=('diff', 'size'))
    rng = np.random.default_rng(SEED)
    num = np.zeros(BOOT)
    den = np.zeros(BOOT)
    for _, z in groups.groupby(level=['suite', 'task'], sort=True):
        weights = rng.multinomial(len(z), np.repeat(1 / len(z), len(z)), size=BOOT)
        num += weights @ z.numerator.to_numpy()
        den += weights @ z.denominator.to_numpy()
    draws = num / den
    return dict(pairs=len(a), original_init_clusters=len(groups),
                success_d=int(a.success_d.sum()), success_p=int(a.success_p.sum()),
                sr_d=float(a.success_d.mean()), sr_p=float(a.success_p.mean()),
                d_only=int(((a.success_d == 1) & (a.success_p == 0)).sum()),
                p_only=int(((a.success_d == 0) & (a.success_p == 1)).sum()),
                difference=float(a['diff'].mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                bootstrap_replicates=BOOT, seed=SEED)


def distribution(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std()),
                p10=float(np.quantile(a, .1)), p50=float(np.quantile(a, .5)),
                p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def gap_text(d):
    if not d or not d['n']:
        return '—'
    return f"{d['mean']:.2f} [{d['p10']:g}/{d['p50']:g}/{d['p90']:g}]"


def cdf_rows(values, meta):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return []
    assert np.all(a == a.astype(int))
    counts = np.bincount(a.astype(int))
    cum = counts.cumsum()
    return [dict(meta, gap=g, count=int(n), total=len(a), cdf=float(cum[g]/len(a)))
            for g, n in enumerate(counts)]


def replace_section(name, section):
    start, end = '<!-- TEST_A_MEASUREMENT_BEGIN -->', '<!-- TEST_A_MEASUREMENT_END -->'
    path = HERE / name
    text = path.read_text()
    assert text.count(start) == text.count(end) == 1, name
    prefix, rest = text.split(start)
    _, suffix = rest.split(end)
    write(name, prefix + start + '\n\n' + section.strip() + '\n\n' + end + suffix)
