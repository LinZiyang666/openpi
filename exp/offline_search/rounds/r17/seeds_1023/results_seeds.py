"""Summarize success-filtered seed experiments; CPU only, never launch experiments.

Run only after the coordinator releases the machine (COMMON.md):
  PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.seeds_1023.results_seeds --workers 8
  ... --verify-existing --workers 8      # replay old runs and compare every available historical cell
  ... --existing-only                   # read the three old summaries, without opening batches
  ... --md-only                         # JSON -> Markdown, stdlib only

SEEDS_INDEX.json is coordinator-owned; see SEEDS_INDEX.example.json. Names are
never guessed. Paths in the index are absolute or relative to that index file.
Missing/invalid inputs are reported, never replaced by old-seed data. Exit 1
means incomplete results or a failed regression; JSON/Markdown still explain why.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import statistics
import traceback

HERE = Path(__file__).resolve().parent
DIMS = ('model', 'benchmark', 'mode', 'long_segment', 'seed')
SEEDS = {'libero': (20261002, 20261003, 20261004), 'mt50': (8, 9, 10)}
MODELS = {'libero': ('dp', 'fastwam', 'cosmos', 'pi05', 'groot'), 'mt50': ('dp', 'fastwam', 'groot', 'pi05')}
LABELS = {'dp': 'DP', 'fastwam': 'Fast-WAM', 'cosmos': 'Cosmos', 'pi05': 'π0.5', 'groot': 'GR00T'}


def segments(model, benchmark):
    return ('default',) if (model, benchmark) in (('cosmos', 'libero'), ('pi05', 'mt50')) else ('off', 'on')


def sizes(model, benchmark):
    return (250, 500, 1000, 1500, 2000, 2500) if benchmark == 'mt50' else (
        (50, 100, 200) if model in ('pi05', 'groot') else (50, 100, 200, 300, 400))


def suites(benchmark):
    return ('mt50',) if benchmark == 'mt50' else ('libero_10', 'libero_spatial')


def cell_key(row):
    return tuple(row[k] for k in (*DIMS, 'suite', 'size'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_index(path):
    index = read(path)
    if index.get('version') != 1 or index.get('example'):
        raise ValueError('Need coordinator-filled SEEDS_INDEX version 1 (remove example=true)')
    references = index['references']

    def absolute(value):
        p = Path(value)
        return str(p if p.is_absolute() else path.parent/p)

    for ref in references.values():
        if ref['pipeline'] not in ('R17', 'R15'):
            raise ValueError('Pure references use R17 runs or explicit R15 root/arm pairs')
        if ref['pipeline'] == 'R17':
            ref['run'] = absolute(ref['run'])
            if not ref.get('groups'):
                raise ValueError('Reference needs explicit pure group names (or control)')
        else:
            if not ref.get('jobs'):
                raise ValueError('VLA reference needs explicit root/arm pairs')
            for job in ref['jobs']:
                job['run'] = absolute(job['run'])
    seen_runs = set()
    for name, batch in index['batches'].items():
        batch['run'] = absolute(batch['run'])
        if batch['run'] in seen_runs:
            raise ValueError(f'{name}: duplicate physical run; put its slices in one batch entry')
        seen_runs.add(batch['run'])
        if batch['pipeline'] not in ('R17', 'R15', 'R16') or not batch['slices']:
            raise ValueError(f'{name}: invalid pipeline/slices')
        for s in batch['slices']:
            m, b, mode = s['model'], s['benchmark'], s['mode']
            if m not in MODELS[b] or mode not in ('full', 'reduced') or s['seed'] not in SEEDS[b][1:]:
                raise ValueError(f'{name}: invalid dimensions {s}')
            if s['long_segment'] not in segments(m, b):
                raise ValueError(f'{name}: unsupported segment {s["long_segment"]}')
            if not s['sizes'] or not set(s['sizes']) <= set(sizes(m, b)) or set(s['suites']) != set(suites(b)):
                raise ValueError(f'{name}: sizes/suites do not match DESIGN.md')
            vla = b == 'libero' and m in ('pi05', 'groot')
            expected_pipeline = ('R15' if mode == 'full' else 'R16') if vla else 'R17'
            if batch['pipeline'] != expected_pipeline:
                raise ValueError(f'{name}: expected {expected_pipeline}')
            selector = s['arms'] if vla else s['groups']
            if not selector or (vla and len(selector) != len(set(selector))):
                raise ValueError(f'{name}: empty or duplicated selector')
            for kind in ('reference', 'long_reference'):
                required = kind == 'reference' or (not vla and s['long_segment'] == 'on')
                if not required:
                    continue
                ref = references[s[kind]]
                for field in ('model', 'benchmark', 'mode', 'gpu'):
                    if s[field] != ref[field]:
                        raise ValueError(f'{name}: {kind} must match {field}: {s[field]} != {ref[field]}')
                if ref['pipeline'] != ('R15' if vla else 'R17'):
                    raise ValueError(f'{name}: wrong reference pipeline')
            if m == 'pi05' and b == 'mt50' and '4090' not in s['gpu']:
                raise ValueError(f'{name}: pi05 MT50 has only 4090 references')
    return index


def worker(item):
    kind, name, payload = item
    try:
        from .results_lib import native, history
        if kind == 'batch':
            value = native.load_batch((name, payload))[1]
        elif kind == 'reference':
            value = native.load_reference(payload)
        else:
            value = history.replay(payload)
        return dict(kind=kind, name=name, value=value)
    except Exception:
        return dict(kind=kind, name=name, error=traceback.format_exc())


def parallel(items, workers):
    if not items:
        return []
    outputs = []
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, item): item[:2] for item in items}
        for future in as_completed(futures):
            out = future.result(); outputs.append(out)
            print(f'{out["kind"]} {out["name"]}: {"ERROR" if "error" in out else "read"}', flush=True)
    return outputs


def new_rows(index, workers):
    from .results_lib import native
    items = [('batch', name, batch) for name, batch in index['batches'].items()]
    used_refs = {s[k] for batch in index['batches'].values() for s in batch['slices']
                 for k in ('reference', 'long_reference') if k in s}
    items += [('reference', name, index['references'][name]) for name in sorted(used_refs)]
    loaded = parallel(items, workers)
    errors = [r for r in loaded if 'error' in r]
    refs = {r['name']: r['value'] for r in loaded if r['kind'] == 'reference' and 'value' in r}
    failed = {r['name'] for r in errors if r['kind'] == 'batch'}
    blocked = {native.key(s) for name in failed for s in index['batches'][name]['slices']}
    lines, data, runs = {}, {}, defaultdict(list)
    for out in sorted(loaded, key=lambda r: (r['kind'], r['name'])):
        if out['kind'] != 'batch' or 'error' in out:
            continue
        for part in out['value']:
            s = part['line']; k = native.key(s)
            line = {f: s[f] for f in (*DIMS, 'gpu', 'reference')}
            if 'long_reference' in s:
                line['long_reference'] = s['long_reference']
            if k in lines and line != lines[k]:
                raise ValueError(f'{k}: GPU/reference changed between off and levels')
            lines[k] = line
            runs[k].append(dict(batch=out['name'], run=index['batches'][out['name']]['run']))
            for cell, arms in part['data'].items():
                for level, episodes in arms.items():
                    native.merge(data.setdefault(k, {}).setdefault(cell, {}).setdefault(level, {}), episodes,
                                 r17=index['batches'][out['name']]['pipeline'] == 'R17')
    rows = []
    for k, line in sorted(lines.items()):
        if k in blocked:
            continue
        try:
            built = native.calculate(line, data[k], refs[line['reference']], refs.get(line.get('long_reference')))
            for row in built:
                row.update(status='complete', source='SEEDS_INDEX.json', runs=runs[k],
                           reference_spec=index['references'][line['reference']])
            rows.extend(built)
        except Exception:
            errors.append(dict(kind='line', name=list(k), error=traceback.format_exc()))
    return rows, errors


def compact(row):
    """Keep measured chain numbers and task decisions; omit duplicated old diagnostics."""
    detail = row['detail']; s = row['summary']
    order = ('ir45', 'ir40', 'ir32', 'ir25', 'off') if row['benchmark'] == 'libero' and row['model'] in ('pi05', 'groot') else (
        'k50', 'k40', 'k30', 'k20', 'k10', 'koff')
    measured = detail.get('measured_levels', {})
    tasks = {}
    for task, d in detail['tasks'].items():
        attempted = [] if not s['tuned'] else list(order[:-1])
        if d.get('first_failure') in attempted:
            attempted = attempted[:attempted.index(d['first_failure'])+1]
        elif s['tuned']:
            attempted.append(order[-1])  # after all levels pass, the terminal off is selected
        tasks[task] = dict(selected=d['selected'], first_failure=d.get('first_failure'),
                           path=[order[-1], *attempted], levels=d['levels'],
                           executed_tolerance_met=d.get('executed_tolerance_met'))
    out = {k: v for k, v in row.items() if k != 'detail'}
    out['chain'] = dict(initial=order[-1], descending_order=list(order[:-1]),
                        no_knob=not s['tuned'], measured_levels=measured, tasks=tasks)
    out['sr_primary'] = s.get('group_mean') if row['benchmark'] == 'mt50' else s.get('sr')
    out['pure_sr_primary'] = s.get('pure_group_mean') if row['benchmark'] == 'mt50' else s.get('pure_sr')
    out['shared_full_library'] = row['benchmark'] == 'libero' and row['size'] == 500
    return out


def expected_grid(existing_only=False):
    for benchmark, models in MODELS.items():
        for model in models:
            for mode in ('full', 'reduced'):
                for segment in segments(model, benchmark):
                    for seed in SEEDS[benchmark][:1] if existing_only else SEEDS[benchmark]:
                        for suite in suites(benchmark):
                            for size in sizes(model, benchmark):
                                yield dict(model=model, benchmark=benchmark, mode=mode, long_segment=segment,
                                           seed=seed, suite=suite, size=size)


def cross_seed(rows):
    cells = defaultdict(dict)
    for row in rows:
        if row['shared_full_library']:
            continue
        k = tuple(row[f] for f in ('model', 'benchmark', 'mode', 'long_segment', 'suite', 'size'))
        cells[k][str(row['seed'])] = row
    out = []
    for k, by_seed in sorted(cells.items()):
        record = dict(zip(('model', 'benchmark', 'mode', 'long_segment', 'suite', 'size'), k))
        metrics = {}
        for name in ('sr_primary', 'ir', 'task_mean', 'pure_sr_primary', 'pure_task_mean', 'pure_ir', 'level_ir'):
            values = {seed: (r.get(name) if name in ('sr_primary', 'pure_sr_primary') else r['summary'].get(name))
                      for seed, r in by_seed.items() if r['status'] == 'complete'}
            complete = len(values) == 3 and all(v is not None for v in values.values())
            v = list(values.values())
            metrics[name] = dict(by_seed=values, mean=statistics.mean(v) if complete else None,
                                 min=min(v) if complete else None, max=max(v) if complete else None,
                                 std=statistics.pstdev(v) if complete else None)
        out.append(dict(record, metrics=metrics, complete=len(by_seed) == 3 and all(r['status'] == 'complete' for r in by_seed.values()),
                        meets_rule={s: r['summary'].get('meets_rule') for s, r in by_seed.items()},
                        missing_seeds=[s for s in SEEDS[k[1]] if str(s) not in by_seed]))
    return out


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def num(value):
    return '—' if value is None else f'{value:.6f}'


def label(row):
    return f'{LABELS[row["model"]]} / {row["suite"]} / {row["mode"]} / {row["long_segment"]} / N={row["size"]}'


def markdown(result):
    out = ['# 新库种子：成功过滤库两步下降', '', f'生成：{result["generated_at"]}；完整：{result["complete"]}。', '',
           'SR：LIBERO 为套件均值；MT50 先列四难度组等权均值，再列任务均值。参照为同卡同步数纯推理。', '',
           '选档和 IR 调用 results_off.py 及原函数：R17 用 4090 profile；VLA 用原生 D5 成本账本。', '',
           'Cosmos LIBERO、π0.5 MT50 只列 default。LIBERO N=500 为旧种子的全集参照，不伪造三个独立种子。', '',
           '只有三个种子都完整且该指标都有值才计算跨种子均值/范围；缺少正式 IR 不会用 level IR 替代。', '',
           '| 格 | 种子 | GPU | SR 组/套件 | SR 任务 | 正式 IR | 纯推理 SR 组/套件 | 纯推理 SR 任务 | 达标 | 最终档位 |',
           '|---|---:|---|---:|---:|---:|---:|---:|---|---|']
    for r in result['rows']:
        s = r['summary']; counts = Counter(s['per_task_levels'].values())
        levels = ' '.join(f'{k}×{v}' for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))
        meets = {True: '是', False: '否', None: '待定'}[s.get('meets_rule')]
        out.append(f'| {label(r)} | {r["seed"]} | {r.get("gpu") or "—"} | {pct(r["sr_primary"])} | {pct(s.get("task_mean"))} | '
                   f'{num(s.get("ir"))} | {pct(r["pure_sr_primary"])} | {pct(s.get("pure_task_mean"))} | {meets} | {levels} |')
    out += ['', '## 跨种子', '', '| 格 | SR 均值 [最小, 最大] | MT50 任务 SR 均值 [最小, 最大] | IR 均值 [最小, 最大] |',
            '|---|---|---|---|']
    for r in result['cross_seed']:
        def span(name, fmt):
            x = r['metrics'][name]
            return f'{fmt(x["mean"])} [{fmt(x["min"])}, {fmt(x["max"])}]'
        out.append(f'| {label(r)} | {span("sr_primary", pct)} | {span("task_mean", pct)} | {span("ir", num)} |')
    out += ['', '## 链：off → 逐档（逐任务）', '']
    for r in result['rows']:
        out += [f'<details><summary>{label(r)} / seed={r["seed"]}</summary>', '',
                '| 档位 | SR 组/套件 | SR 任务 | 正式 IR |', '|---|---:|---:|---:|']
        chain = r['chain']
        for level in [chain['initial'], *chain['descending_order']]:
            st = chain['measured_levels'].get(level, {})
            out.append(f'| {level} | {pct(st.get("group_mean") if r["benchmark"] == "mt50" else st.get("sr"))} | '
                       f'{pct(st.get("task_mean"))} | {num(st.get("ir"))} |')
        out += ['', '| 任务 | 链 | 最终档 | 首次失败 |', '|---|---|---|---|']
        for task, d in sorted(chain['tasks'].items()):
            out.append(f'| {task} | {" → ".join(d["path"])} | {d["selected"]} | {d["first_failure"] or "—"} |')
        out += ['', '</details>', '']
    if result['missing'] or result['errors'] or result['missing_ir']:
        out += ['## 未完成 / 错误', '']
        out += [f'- 缺格：{label(r)} / seed={r["seed"]}' for r in result['missing']]
        out += [f'- 缺正式 IR：{label(r)} / seed={r["seed"]}' for r in result['missing_ir']]
        out += [f'- {e["kind"]} {e["name"]}: {e["error"].strip().splitlines()[-1]}' for e in result['errors']]
    return '\n'.join(out)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=HERE/'SEEDS_INDEX.json')
    p.add_argument('--json', type=Path, default=HERE/'RESULTS_SEEDS.json')
    p.add_argument('--md', type=Path, default=HERE/'RESULTS_SEEDS.md')
    p.add_argument('--workers', type=int, default=8)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--md-only', action='store_true')
    group.add_argument('--existing-only', action='store_true')
    group.add_argument('--verify-existing', action='store_true')
    p.add_argument('--regression-json', type=Path, default=HERE/'REGRESSION_SEEDS.json')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.md_only:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(read(args.json)))
        return
    from .results_lib import history
    old, specs = history.read_existing(replay=args.verify_existing)
    if args.verify_existing:
        outputs = parallel([('regression', ':'.join(map(str, tuple(s['line'][k] for k in DIMS))), s) for s in specs], args.workers)
        passed = all('error' not in o and not o['value']['differences'] for o in outputs)
        write(args.regression_json, dict(passed=passed, checked=sum(o.get('value', {}).get('checked', 0) for o in outputs),
                                         tolerance='IR/SR floats: abs/rel 1e-12; counts, selections and chains exact', results=outputs))
        print(f'Regression {"PASS" if passed else "FAIL"}: {args.regression_json}', flush=True)
        raise SystemExit(0 if passed else 1)
    rows = [compact(r) for r in old]; errors = []
    if not args.existing_only:
        index = read_index(args.index)
        fresh, errors = new_rows(index, args.workers)
        rows += [compact(r) for r in fresh]
    keyed = {}
    for r in rows:
        k = cell_key(r)
        if k in keyed:
            raise ValueError(f'Duplicate seed cell {k}; historical seeds must not be overwritten')
        keyed[k] = r
    missing = [r for r in expected_grid(args.existing_only) if cell_key(r) not in keyed or keyed[cell_key(r)]['status'] != 'complete']
    missing_ir = [{k: r[k] for k in (*DIMS, 'suite', 'size')} for r in rows
                  if r['summary'].get('ir') is None or r['summary'].get('pure_ir') is None]
    rows.sort(key=cell_key)
    result = dict(version=1, generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  complete=not (missing or errors or missing_ir), existing_only=args.existing_only,
                  index=None if args.existing_only else str(args.index), sources=[str(p) for p in history.SOURCES],
                  rows=rows, cross_seed=cross_seed(rows), missing=missing, missing_ir=missing_ir, errors=errors)
    write(args.json, result)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(markdown(result))
    print(f'Wrote {len(rows)} cells; missing={len(missing)}, errors={len(errors)}, missing IR={len(missing_ir)}', flush=True)
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
