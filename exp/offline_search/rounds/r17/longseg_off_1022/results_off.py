"""Long segment OFF (commit 5), success-filtered library: final per-cell SR / IR next to the ON study.

CPU only, read only. The numbers come from the ON study's own code, pointed at the OFF runs and registry:
  R17 (DP / Fast-WAM LIBERO + MT50, GR00T MT50): the filtered-arm body of longseg_1017.analyze._main --
    analyze.lane_data / reference / stats / decision_summary / prices / whole_segment_result,
    knob.descent.analyze_mt50 (MT50) and formal_a_analysis.choose (LIBERO) -- over longseg_off_1022
    CAMPAIGNS.json + FORMAL_QUEUE.json; the reference is longseg_off_1022/REFERENCES.json at
    model:benchmark:mode:build.card(run). FINAL_CHAINS_OFF.json overrides run dirs (pod copies, retries)
    and reference entries.
  pi0.5 / GR00T LIBERO: the q3_port_1020.results.main body -- analysis.load_job (its long-segment-OFF
    reader shim) with formal_a_analysis.legacy_data = results.ledger, formal_a_analysis.choose,
    results.stats -- over filter_failed_knob/longseg_off_1022/QUEUE.json; the reference is that study's
    build.references (H100 same-step C5 pure, fa_lo_*_p5_q).
A line is aggregated only when every run it needs (knob batch, levels, reference) has all planned episodes
accepted; otherwise it is pending with accepted/expected counts. Re-run to refresh. ON = the
success-filtered arm of longseg_1017/RESULTS_1022.json and q3_port_1020/RESULTS_1022.json.

    PYTHONPATH=.:src nice -n 10 .venv/bin/python -m exp.offline_search.rounds.r17.longseg_off_1022.results_off
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from unittest.mock import patch

from exp.offline_search.rounds.r17.cache import analyze_cache as cache
from exp.offline_search.rounds.r17.filter_failed_knob import formal_a as f, formal_a_analysis as fa
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, cells as library_cells
from exp.offline_search.rounds.r17.knob import common as c, descent
from exp.offline_search.rounds.r17.longseg_1017 import analyze as an
from exp.offline_search.rounds.r17.longseg_1017.io_budget import dump
from . import build as ob

HERE = Path(__file__).resolve().parent
R17 = HERE.parent
VLA = R17/'filter_failed_knob'/'longseg_off_1022'
ON_R17 = R17/'longseg_1017'/'RESULTS_1022.json'
ON_VLA = R17/'filter_failed_knob'/'q3_port_1020'/'RESULTS_1022.json'
CHAINS = HERE/'FINAL_CHAINS_OFF.json'
ORDER = ('k50', 'k40', 'k30', 'k20', 'k10', 'koff')
SHOW = ('pure', 'k50', 'k40', 'k30', 'k20', 'k10', 'koff', 'ir45', 'ir40', 'ir32', 'ir25', 'off')
MODEL = {'dp': 'DP', 'fastwam': 'Fast-WAM', 'groot': 'GR00T', 'pi05': 'π0.5'}
MODE = {'reduced': '减步', 'full': '全步'}
SUITE = {'libero_10': '长任务 l10', 'libero_spatial': '空间 spatial'}
RELAXATIONS = [
    'π0.5/GR00T loaders: q3_port_1020/analysis.py as it is (imported, not edited): r16.analyze.require skips the '
    '"Changed judge" and "Changed client/policy protocol" checks, and its Reader fills vision=True and a policy call\'s '
    'miss_k (the shard\'s startup miss_steps) on decisions logged without them (long segment OFF serves without '
    '--os-blind, which is what logs them). Nothing else is relaxed; no shared loader was edited.',
    'R17 whole_segment_result(selection, data, target, long_pure, ...): long_pure is the same-card C5 pure reference '
    '(target) itself. ON executes a task frozen at pure with fresh pure+long-segment (C10) episodes because its system '
    'runs C10; the OFF system runs C5 segments, so its pure fallback is the C5 reference. The labels '
    '"pure_long" in its output are renamed accordingly; native_length_fallback_diagnostic (identical to the final '
    'here) is dropped.',
    'R17 measured_levels carry no RECALIBRATION.json predictions (those are the ON C10 decision-clock fits; OFF '
    'reuses the ON periodic_gap values without recalibration). They are descriptive only in ON.',
    'Run dirs and references: FINAL_CHAINS_OFF.json maps canonical jobs to the dirs that hold their data and '
    'overrides two REFERENCES.json entries (listed there with reasons).',
]


# ----------------------------------------------------------------------------------------- progress
def r17_progress(run):
    """Accepted terminal episodes (analyze_cache.accepted_terminal over the resume chain) vs the episode plan."""
    run = Path(run)
    out = dict(run=str(run), receipt=None, execution_host=None, accepted=0, expected=None, complete=False)
    if (run/'RECEIPT.json').exists():
        receipt = read(run/'RECEIPT.json')
        out.update(receipt=receipt.get('status'), execution_host=receipt.get('execution_host'))
    plan = run/'FULL_EPISODE_PLAN.json'
    if not plan.exists():
        plan = run/'EPISODE_PLAN.json'
    if not plan.exists():
        return dict(out, note='not prepared')
    expected = {t['task_uid'] for t in read(plan)['tasks']}
    sources = cache.Sources()
    journal = cache.accepted_terminal(cache.evidence_rows(cache.resume_segments(run.resolve(), sources),
                                                          'journal.jsonl', sources))
    accepted = len(expected & journal.keys())
    return dict(out, accepted=accepted, expected=len(expected), complete=bool(expected) and accepted == len(expected))


def journal_uids(path):
    uids = set()
    if path.exists():
        for line in path.open('rb'):
            if line.strip():
                r = json.loads(line)
                if r.get('accepted') and not r.get('error') and r.get('status') in ('done', 'failed'):
                    uids.add(r['task_uid'])
    return uids


def vla_progress(unit):
    """R15 root: every arm has runs/<arm>/summary.json; R16 group: DONE.json. Counts from the journals."""
    parts = []
    for job in unit['jobs']:
        root = Path(job['run'])
        if job['pipeline'] == 'R15':
            arms = read(root/'arms.json') if (root/'arms.json').exists() else []
            done = [a for a in arms if (root/'runs'/a['arm']/'summary.json').exists()]
            accepted = sum(len(journal_uids(root/'runs'/a['arm']/'client/journal.jsonl')) for a in arms)
            parts.append(dict(name=job['name'], accepted=accepted, expected=job['episodes'], arms_done=f'{len(done)}/{len(arms)}',
                              complete=bool(arms) and len(done) == len(arms) and accepted == job['episodes']))
        else:
            accepted = len(journal_uids(root/'journal.jsonl'))
            done = (root/'DONE.json').exists()
            parts.append(dict(name=job['name'], accepted=accepted, expected=job['episodes'], done=done,
                              complete=done and accepted == job['episodes']))
    return dict(accepted=sum(p['accepted'] for p in parts), expected=sum(p['expected'] for p in parts),
                complete=bool(parts) and all(p['complete'] for p in parts), jobs=parts)


# ------------------------------------------------------------------------------------------- lines
def r17_lines(chains, registry):
    jobs = list(read(HERE/'CAMPAIGNS.json')['jobs'])
    queue = read(HERE/'FORMAL_QUEUE.json')['units'] if (HERE/'FORMAL_QUEUE.json').exists() else []
    names = {j['name'] for j in jobs}
    jobs += [j for u in queue for j in u['jobs'] if j['name'] not in names]  # as analyze._main
    runs = chains.get('runs', {})
    lines = {}
    for j in jobs:
        if j.get('stage') not in ('descent', 'off', 'levels'):
            continue
        line = lines.setdefault((j['model'], j['benchmark'], j['mode']), dict(
            model=j['model'], benchmark=j['benchmark'], mode=j['mode'], pipeline='R17', jobs=[], levels_unit=None))
        line['jobs'].append(dict(name=j['name'], stage=j['stage'], run=runs.get(j['name'], j['run']), canonical_run=j['run']))
    for u in queue:
        if u['stage'] == 'levels':
            lines[u['model'], 'libero', u['mode']]['levels_unit'] = dict(name=u['name'], status=u['status'],
                                                                         selected_cells=u.get('selected_cells'))
    for line in lines.values():
        knob = next(j for j in line['jobs'] if j['stage'] in ('descent', 'off'))
        line['gpu'] = ob.card(knob['run']) if (Path(knob['run'])/'campaign.json').exists() else 'H100'
        key = f"{line['model']}:{line['benchmark']}:{line['mode']}:{line['gpu']}"
        ref = registry.get(key)
        parts = [dict(part=j['stage'], name=j['name'], **r17_progress(j['run'])) for j in line['jobs']]
        unit = line['levels_unit']
        if line['benchmark'] == 'libero' and (unit is None or unit['status'] not in ('BUILT', 'SKIP_OFF_SUFFICIENT')):
            parts.append(dict(part='levels', name=unit and unit['name'], status=unit and unit['status'], accepted=0,
                              expected=None, complete=False, note='levels not built yet'))
        if ref is None:
            parts.append(dict(part='reference', name=key, accepted=0, expected=None, complete=False,
                              note='same-card C5 pure reference not registered'))
        else:
            parts.append(dict(part='reference', name=Path(ref['run']).name, key=key, **r17_progress(ref['run'])))
        line.update(reference_key=key, reference=ref, parts=parts, complete=all(p['complete'] for p in parts),
                    knob_off_runs=[j['run'] for j in line['jobs'] if j['stage'] in ('descent', 'off')
                                   and next(p for p in parts if p['name'] == j['name'])['complete']])
    return lines


def vla_lines():
    from exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022 import build as vb
    units = read(VLA/'QUEUE.json')['units']
    lines = {}
    for model in ('pi05', 'groot'):
        for mode in ('reduced', 'full'):
            off = next(u for u in units if (u['model'], u['mode'], u['stage']) == (model, mode, 'off'))
            ref = next(u for u in units if (u['model'], u['mode'], u['stage'], u.get('segment')) == (model, mode, 'pure', 'c5'))
            levels = next((u for u in units if (u['model'], u['mode'], u['stage']) == (model, mode, 'levels')), None)
            decision = VLA/'off_decisions'/f'{model}_{mode}_filtered.json'
            off_part = dict(part='off', name=off['name'], **vla_progress(off))
            parts = [off_part]
            knob = [off]
            if levels is not None:
                parts.append(dict(part='levels', name=levels['name'], **vla_progress(levels)))
                knob.append(levels)
            elif not (decision.exists() and all(r['status'] == 'skip' for r in read(decision)['rows'])):
                parts.append(dict(part='levels', name=vb.name(model, mode, 'levels'), accepted=0, expected=None,
                                  complete=False, note='levels not built yet'))
            parts.append(dict(part='reference', name=ref['name'], **vla_progress(ref)))
            lines[model, 'libero', mode] = dict(
                model=model, benchmark='libero', mode=mode, pipeline='R15' if mode == 'full' else 'R16', gpu='H100',
                units=knob, reference_unit=ref['name'], reference_arms=None, parts=parts,
                complete=all(p['complete'] for p in parts), off_decision=str(decision) if decision.exists() else None,
                knob_off_complete=off_part['complete'])
    return lines


# ------------------------------------------------------------------------------- loading (pool)
def tally_r17(t, row):
    kind = row.get('_kind') or 'decision'
    if kind == 'decision':
        t['decisions'] += 1
        t['source:'+str(row.get('source'))] += 1
        t['not_segment_head'] += row.get('segment_head') is not True
        t['commit_horizon:'+str(row.get('commit_horizon'))] += 1
        t['miss_mode:'+str(row.get('miss_mode'))] += 1
        t['periodic_gap_set'] += row.get('periodic_gap') is not None
        t['stall_extra_look'] += bool(row.get('stall_extra_look'))
        if row.get('source') == 'policy':
            if row.get('stall_call'):
                t['policy_stall_call'] += 1
            elif row.get('miss_mode') is None:  # recipe off: a task with no successful library episode
                t['policy_recipe_off'] += 1
                t[f"recipe_off_task:{str(row.get('arm')).partition('_')[0]}:{row.get('task_name')}"] += 1
            else:
                t['policy_other'] += 1
    elif kind == 'execution_ack':
        t['acks'] += 1
        t['acks_over_5_controls'] += (row.get('executed_count') or 0) > 5


def tally_vla(t, d):
    t['decisions'] += 1
    t['source:'+str(d.get('source', d.get('src')))] += 1
    t['logged_vision'] += 'vision' in d  # only the --os-blind (long segment) path logs it
    if d.get('hit') is False:
        t['model_calls'] += 1
        extras, r14 = d.get('extras') or {}, d.get('r14') or {}
        if extras.get('os_q3_call') == 1:
            t['call_q3_stall'] += 1
        elif r14.get('knob'):
            t['call_knob'] += 1
        else:
            t['call_other:'+str(d.get('judge'))] += 1


def r17_lane(root):
    """analyze.lane_data(root); the accepted per_step rows it reads are tallied for the decision-log check."""
    audit = {}
    native = cache.evidence_rows

    def tapped(segments, name, sources):
        for row in native(segments, name, sources):
            # LIBERO rows carry cache_group; MetaWorld rows only the arm (n<size>_<group>).
            group = row.get('cache_group') or str(row.get('arm') or '').partition('_')[2]
            if name == 'per_step.jsonl' and row.get('accepted') is True and group:
                tally_r17(audit.setdefault(group, Counter()), row)
            yield row
    with patch.object(cache, 'evidence_rows', tapped):
        data = an.lane_data(root)
    return dict(data=data, audit={g: dict(v) for g, v in audit.items()})


def vla_load(kind, payload):
    """q3_port_1020 results._task with the OFF references; server decision rows are tallied as they are read."""
    from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import analysis, results
    from exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022 import build as vb
    audit = {}
    native = analysis._jsonl

    def tapped(self, path):
        path = Path(path)
        tap = path.name.startswith('decisions_') and path.parent.name.startswith('server_')
        for d in native(self, path):
            if tap and d.get('ev') == 'dec':
                tally_vla(audit.setdefault(path.parent.parent.name, Counter()), d)
            yield d
    with patch.object(fa, 'legacy_data', results.ledger), patch.object(analysis, '_jsonl', tapped):
        if kind == 'vla_job':
            data, used = analysis.load_job(payload), None
        else:
            data, used = vb.references(*payload)
    return dict(data=data, used=used, audit={a: dict(v) for a, v in audit.items()})


def load(item):
    try:
        kind, payload = item
        return r17_lane(payload) if kind == 'r17' else vla_load(kind, payload)
    except Exception:  # reported per item; the line is then not aggregated
        return {'__error__': traceback.format_exc()}


def empty_tasks(model):
    """(n<size>, task) whose success-filtered library has no episode (filter_failed CELLS.json)."""
    return {f'n{size}:{task}' for (m, suite, size), cell in library_cells().items() if m == model
            for task in cell['empty_tasks']}


# ------------------------------------------------------------------------------------ aggregation
def r17_rows(line, results_dir):
    """The filtered-arm body of longseg_1017.analyze._main for one (model, benchmark, mode)."""
    model, benchmark, mode, gpu = line['model'], line['benchmark'], line['mode'], line['gpu']
    data = {}
    for j in line['jobs']:
        for g in read(Path(j['run'])/'assets/CACHE_STUDY.json')['groups']:
            data.setdefault((g['suite'], g['library_size']), {})
        for (_, suite, size), arms in an.lane_data(j['run']).items():
            for level, episodes in arms.items():
                c.merge(data.setdefault((suite, size), {}).setdefault(level, {}), episodes)
    with patch.object(an, 'registry', lambda: {line['reference_key']: line['reference']}):
        refs = an.reference(model, benchmark, mode, gpu)
        entry = an.reference_entry(model, benchmark, mode, gpu)
    selections = {}
    if benchmark == 'mt50':
        native = descent.analyze_mt50({(an.LABELS[model], s, n): v for (s, n), v in data.items()},
            {(an.LABELS[model], s): v for s, v in refs.items()}, results_dir/f'{model}_{mode}_filtered', mode,
            control_overrides=[an.LABELS[model]+'='+Path(entry['run']).name],
            prices=lambda label, suite: an.prices(label, suite, gpu), summary=an.decision_summary)
        for r in native['cells']:
            chosen = {}
            for task, d in r['tasks'].items():
                level = d['selected']
                pool = refs.get(r['suite'], {}) if level == 'pure' else data[r['suite'], r['size']].get(level, {})
                chosen.update({k: v for k, v in pool.items() if k[1] == task})
            selections[r['suite'], r['size']] = (r, chosen)
    else:
        for (suite, size), arms in data.items():
            d = fa.choose(arms, refs.get(suite, {}), ORDER)
            selections[suite, size] = (dict(an.stats(d['data'], model, mode, suite, gpu), tasks=d['tasks'],
                                            no_knob=d['no_knob'], complete=d['complete']), d['data'])
    libs = library_cells()
    runs = [dict(name=j['name'], run=j['run'], canonical_run=j['canonical_run']) for j in line['jobs']]
    rows = []
    for suite, size in sorted(selections):
        target = refs.get(suite, {})
        # OFF runs C5 segments: a task frozen at pure executes the same-card C5 pure reference (long_pure = target).
        final, _ = an.whole_segment_result(*selections[suite, size], target, target, model, mode, suite, gpu)
        final.pop('native_length_fallback_diagnostic', None)
        final['fallback'] = 'pure: the same-card same-step C5 pure reference (the OFF system executes C5 segments)'
        for t in final['tasks'].values():
            if t.get('executed_fallback'):
                t['executed_fallback'] = 'pure_c5_reference'
        measured = data.get((suite, size), {})
        final.update(runs=runs, reference_run=entry['run'], reference_key=line['reference_key'],
                     native_pure_target=an.stats(target, model, mode, suite, gpu))
        final['measured_levels'] = {level: dict(an.stats(measured.get(level, {}), model, mode, suite, gpu),
                                                unused_off_sufficient=bool(final.get('no_knob')) and level != 'koff')
                                    for level in ORDER}
        for task, selection in final['tasks'].items():
            for level, e in selection.get('levels', {}).items():
                pool = target if level == 'pure' else measured.get(level, {})
                metric = an.stats({k: v for k, v in pool.items() if k[1] == task}, model, mode, suite, gpu)
                e.update(level_ir=metric['level_ir'], latency_ir=metric['latency_ir'], decision_heads=metric['decision_heads'])
        info = libs[model, suite, size]
        rows.append(dict(model=model, benchmark=benchmark, mode=mode, suite=suite, size=size, gpu=gpu, arm='filtered',
                         long_segment='off', filtered=final, filtered_episodes_kept=info['kept'],
                         collected=info['collected'], empty_tasks=info['empty_tasks'],
                         expected_outcomes_per_arm=2500 if suite == 'mt50' else 500))
    return rows


def vla_rows(line, loaded, refs):
    """The q3_port_1020.results.main row construction for one (model, mode), filtered arm."""
    from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import results
    model, mode = line['model'], line['mode']
    data, runs = {}, []
    for unit, job, out in loaded:
        runs.append(dict(unit=unit['name'], name=job['name'], run=job['run'], stage=unit['stage']))
        for cell, levels in out.items():
            for level, episodes in levels.items():
                data.setdefault(cell, {}).setdefault(level, {}).update(episodes)
    rows = []
    for suite, size in sorted(data):
        arms, pure = data[suite, size], refs.get(suite, {})
        d = fa.choose(arms, pure, f.VLA_LEVELS)
        rows.append(dict(model=model, benchmark='libero', mode=mode, arm='filtered', long_segment='off', suite=suite,
                         size=size, gpu='H100', runs=runs, reference_arms=line['reference_arms'],
                         pure=results.stats(pure, model, mode, suite), final=results.stats(d['data'], model, mode, suite),
                         no_knob=d['no_knob'], tuned=d['tuned'], complete=d['complete'], grid_complete=d['grid_complete'],
                         per_task_levels={t: x['selected'] for t, x in d['tasks'].items()}, tasks=d['tasks'],
                         measured_levels={level: results.stats(v, model, mode, suite) for level, v in arms.items()}))
    return rows


# ------------------------------------------------------------------------- compact cell summaries
def vla_meets(row):
    if row['no_knob']:
        return True
    tasks = row['tasks'].values()
    return all(t['selected'] == 'pure' or (t['selected'] and t['levels'][t['selected']]['passes']['4'] is True) for t in tasks)


def summary_r17(row):
    x = row['filtered']
    t = x.get('native_pure_target') or row.get('native_pure_target') or {}
    return dict(n=x.get('n'), successes=x.get('successes'), sr=x.get('sr'), group_mean=x.get('group_mean'),
                task_mean=x.get('task_mean'), level_ir=x.get('level_ir'), ir=x.get('ir'), pure_n=t.get('n'),
                pure_successes=t.get('successes'), pure_sr=t.get('sr'), pure_group_mean=t.get('group_mean'),
                pure_task_mean=t.get('task_mean'), pure_ir=t.get('ir'), tuned=not x.get('no_knob'),
                meets_rule=True if x.get('no_knob') else x.get('tolerance_all_tasks'),
                levels=dict(Counter(v['selected'] for v in x['tasks'].values())),
                per_task_levels={k: v['selected'] for k, v in x['tasks'].items()})


def summary_vla(row):
    x, p = row['final'], row['pure']
    return dict(n=x.get('n'), successes=x.get('successes'), sr=x.get('sr'), level_ir=x.get('level_ir'), ir=x.get('ir'),
                pure_n=p.get('n'), pure_successes=p.get('successes'), pure_sr=p.get('sr'), pure_ir=p.get('ir'),
                pure_level_ir=p.get('level_ir'), tuned=not row['no_knob'], meets_rule=vla_meets(row),
                levels=dict(Counter(row['per_task_levels'].values())), per_task_levels=row['per_task_levels'])


def on_values():
    out = {}
    for r in read(ON_R17)['rows']:
        if r['model'] in ('dp', 'fastwam', 'groot'):
            out[r['model'], r['benchmark'], r['mode'], r['suite'], r['size']] = dict(summary_r17(r), gpu=r['gpu'])
    for r in read(ON_VLA)['rows']:
        if r['arm'] == 'filtered':
            out[r['model'], 'libero', r['mode'], r['suite'], r['size']] = dict(summary_vla(r), gpu=None)
    return out


# ------------------------------------------------------------------------------------------ markdown
def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'


def num(x):
    return '—' if x is None else f'{x:.3f}'


def span(values, scale=1., digits=1):
    v = [x*scale for x in values if x is not None]
    if not v:
        return '—'
    lo, hi = f'{min(v):.{digits}f}', f'{max(v):.{digits}f}'
    return lo if lo == hi else f'{lo}–{hi}'


def levels_text(s):
    if not s['tuned']:
        return '旋钮关已达纯推理，未调'
    order = {k: i for i, k in enumerate(SHOW)}
    return ' '.join(f'{k}×{v}' for k, v in sorted(s['levels'].items(), key=lambda kv: order.get(kv[0], 99)))


def line_label(line):
    bench = 'MetaWorld' if line['benchmark'] == 'mt50' else 'LIBERO'
    return f"成功过滤库 · {MODEL[line['model']]} · {bench} · {MODE[line['mode']]}"


def cell_label(line, suite, size):
    return line_label(line) + (f' · 库 {size}' if suite == 'mt50' else f' · {SUITE[suite]} · 库 {size}')


def part_text(p):
    what = {'descent': '旋钮逐档（一批）', 'off': '旋钮关', 'levels': '各档', 'reference': 'C5 同卡纯推理参照'}[p['part']]
    if p.get('note') and not p.get('expected'):
        notes = {'levels not built yet': '各档尚未建', 'same-card C5 pure reference not registered': '参照未登记',
                 'not prepared': '未 prepare'}
        return f"{what} {p.get('name') or ''}：{notes.get(p['note'], p['note'])}"
    count = f"{p['accepted']:,}/{p['expected']:,}" if p.get('expected') else f"{p['accepted']:,}/?"
    extra = f"，收据 {p['receipt']}" if p.get('receipt') else ''
    return f"{what} {p['name']}：{count}{extra}{'' if p['complete'] else '（未完）'}"


def markdown(result):
    lines_by = result['lines']
    rows = result['rows']
    out = [f"# 长执行段关 · 成功过滤库 · 旋钮下降：最终 SR / IR（{result['generated_at'][:16].replace('T', ' ')} UTC 生成）", '',
           '由 `longseg_off_1022/results_off.py` 生成，可重跑刷新；数字全部出自长段开 RESULTS_1022 用的同一套函数。', '',
           '## 口径', '',
           '- **R17（DP、Fast-WAM 的 LIBERO 与 MT50，GR00T 的 MT50）**：`longseg_1017/analyze.py` 里 `_main` 对成功过滤库这一臂的那段，原样调用 '
           '`lane_data`、`reference`、`stats`、`decision_summary`、`prices`、`whole_segment_result`，MT50 选档用 `knob/descent.analyze_mt50`，'
           'LIBERO 用 `formal_a_analysis.choose`。批次取 `longseg_off_1022/CAMPAIGNS.json` + `FORMAL_QUEUE.json`，实际数据目录按 `FINAL_CHAINS_OFF.json`；'
           '参照取 `longseg_off_1022/REFERENCES.json` 里 `模型:基准:步数:卡`（卡 = `build.card(run)`：pod 副本是 H200 MIG 1g.18gb，其余 H100），同样被 `FINAL_CHAINS_OFF.json` 覆盖。',
           '- **π0.5 / GR00T LIBERO**：`q3_port_1020/results.py` 的 `main` 那段：`analysis.load_job`（含其长段关读取补丁）+ `legacy_data = results.ledger`，'
           '`formal_a_analysis.choose`，`results.stats`。批次取 `filter_failed_knob/longseg_off_1022/QUEUE.json`；参照是该研究的 `build.references`（H100 同步数 C5 纯推理 `fa_lo_*_p5_q`）。',
           '- **选档规则**（未改）：旋钮关整格成功数（MT50 为组均值）不低于同卡同步数纯推理就整格用旋钮关；否则逐任务从高档往下，首次比纯推理少 3 局及以上就冻结上一档（2 局容差）。'
           '任务冻结在 pure 时，执行的就是这份 C5 纯推理参照的局（长段关系统本身就是每 5 个控制决策一次）。',
           '- **SR**：LIBERO 每套件 500 局；MetaWorld 先报四难度组等权均值，再报逐任务均值。**纯推理**：长段关 = 同卡同步数 C5 纯推理；长段开 = RESULTS_1022 里的同步数原长度纯推理。',
           '- **level IR**：R17 = 模型调用段头 / 全部段头；π0.5/GR00T = 账本里的模型调用 / 有视觉的决策。**正式 IR**：R17 = 4090 计时档案计价（`analyze.prices`，不重标），'
           'π0.5/GR00T = R15/R16 原生每决策成本账本（D5）。括号里是纯推理参照的同口径 IR。',
           '- **完整性**：一条线要等它的旋钮批次、各档、参照全部局都 accepted 才汇总；没完的只报进度（已收/计划局数），不出部分结果。', '',
           '## 每条线的进度', '',
           '| 线 | 状态 | 组成（已收 / 计划局数） |', '|---|---|---|']
    for key, line in lines_by.items():
        state = '完成' if line['status'] == 'complete' else ('读取出错' if line['status'] == 'error' else '未完（pending）')
        cls = line.get('off_vs_pure_classification')
        cls = f"；旋钮关对纯推理判格（各档据此建）：{' '.join(f'{k}×{v}' for k, v in sorted(cls.items()))}" if cls else ''
        out.append(f"| {line['label']} | {state} | {'；'.join(part_text(p) for p in line['parts'])}{cls} |")
    done = [dict(l, model=k.split(':')[0], benchmark=k.split(':')[1], mode=k.split(':')[2])
            for k, l in lines_by.items() if l['status'] == 'complete']
    out += ['', '## 各组汇总（完成的线；范围是各库大小的最小–最大）', '',
            'SR 一列：MetaWorld 是四难度组等权均值，LIBERO 是 500 局成功率。「达标」：未调的格看整格不低于纯推理；调过的格看执行后每个任务比纯推理少不超过 2 局。', '',
            '| 组 | 格数 | SR 关（%） | SR 开（%） | 纯推理 SR 关（C5 同卡） | 纯推理 SR 开 | 调过的格 关/开 | 达标的格 关/开 | level IR 关 | level IR 开 | 正式 IR 关（纯推理） | 正式 IR 开（纯推理） |',
            '|---|---:|---|---|---|---|---|---|---|---|---|---|']
    for line in done:
        cells = [r for r in rows if (r['model'], r['benchmark'], r['mode']) == (line['model'], line['benchmark'], line['mode'])]
        off = [r['off'] for r in cells]
        on = [r['on'] for r in cells if r['on']]
        sr = 'group_mean' if line['benchmark'] == 'mt50' else 'sr'
        psr = 'pure_group_mean' if line['benchmark'] == 'mt50' else 'pure_sr'
        out.append(f"| {line['label']} | {len(cells)} | {span([s[sr] for s in off], 100)} | {span([s[sr] for s in on], 100)} | "
                   f"{span([s[psr] for s in off], 100)} | {span([s[psr] for s in on], 100)} | "
                   f"{sum(s['tuned'] for s in off)}/{sum(s['tuned'] for s in on) if on else '—'} | "
                   f"{sum(bool(s['meets_rule']) for s in off)}/{sum(bool(s['meets_rule']) for s in on) if on else '—'} | "
                   f"{span([s['level_ir'] for s in off], digits=3)} | {span([s['level_ir'] for s in on], digits=3)} | "
                   f"{span([s['ir'] for s in off], digits=3)}（{span([s['pure_ir'] for s in off], digits=3)}） | "
                   f"{span([s['ir'] for s in on], digits=3)}（{span([s['pure_ir'] for s in on], digits=3)}） |")
    for bench, title in (('mt50', 'MetaWorld（MT50，seed 7，50 任务 × 50 起点）'), ('libero', 'LIBERO（A 集，每套件 10 任务 × 50 起点）')):
        sel = [r for r in rows if r['benchmark'] == bench]
        if not sel:
            continue
        out += ['', f'## 逐格：{title}', '']
        if bench == 'mt50':
            out += ['| 实验 | SR 关（组 / 任务） | SR 开（组 / 任务） | 纯推理 关（组 / 任务） | 纯推理 开（组 / 任务） | level IR 关 / 开 | 正式 IR 关（纯推理） | 正式 IR 开（纯推理） | 关：逐任务选档 | 开：逐任务选档 |',
                    '|---|---:|---:|---:|---:|---|---|---|---|---|']
        else:
            out += ['| 实验 | SR 关 | SR 开 | 纯推理 SR 关 | 纯推理 SR 开 | level IR 关 / 开 | 正式 IR 关（纯推理） | 正式 IR 开（纯推理） | 关：逐任务选档 | 开：逐任务选档 |',
                    '|---|---:|---:|---:|---:|---|---|---|---|---|']
        for r in sel:
            s, o = r['off'], r['on'] or {}
            if bench == 'mt50':
                cols = [f"{pct(s['group_mean'])} / {pct(s['task_mean'])}", f"{pct(o.get('group_mean'))} / {pct(o.get('task_mean'))}",
                        f"{pct(s['pure_group_mean'])} / {pct(s['pure_task_mean'])}", f"{pct(o.get('pure_group_mean'))} / {pct(o.get('pure_task_mean'))}"]
            else:
                cols = [pct(s['sr']), pct(o.get('sr')), pct(s['pure_sr']), pct(o.get('pure_sr'))]
            out.append(f"| {r['label']} | {' | '.join(cols)} | {num(s['level_ir'])} / {num(o.get('level_ir'))} | "
                       f"{num(s['ir'])} ({num(s['pure_ir'])}) | {num(o.get('ir'))} ({num(o.get('pure_ir'))}) | {levels_text(s)} | "
                       f"{levels_text(o) if o else '—'} |")
    out += ['', '## 决策日志核对（旋钮关组）', '',
            '只看旋钮关（R17 的 koff 组 / π0.5、GR00T 的 off 臂），读的是汇总时同一遍读过的逐决策记录（R17：accepted 的 per_step 行；π0.5/GR00T：服务端 decisions_*.jsonl 的全部 dec 行，含被拒尝试）。'
            '长段关应当是：每个决策都是段头、每段最多执行 5 个控制、没有 policy_tail（盲尾）、不记 vision（只有 --os-blind 路径才记）；旋钮关时模型调用只来自 Q3 停滞门。', '']
    r17_audit = [a for a in result['decision_log_audit'] if a['pipeline'] == 'R17']
    vla_audit = [a for a in result['decision_log_audit'] if a['pipeline'] != 'R17']
    if r17_audit:
        out += ['「库空任务调用」：成功过滤库里该任务一局成功的都没有（filter_failed CELLS.json 的 empty_tasks），这些任务 recipe 关闭（miss_mode 空），每次决策都调模型，'
                '与长段开同一机制；最后一列核对这些调用是否全部落在 empty_tasks 上。', '',
                '| 组（R17） | 决策行 | 非段头 | commit_horizon | 执行 >5 控制的 ack | 模型调用 | 其中停滞门 stall_call | 库空任务调用 | 其他调用 | periodic_gap 非空 | miss_mode | 库空调用只在 empty_tasks 上 |',
                '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
        for a in r17_audit:
            t = a['counts']
            ch = ' '.join(f"{k.split(':', 1)[1]}×{v:,}" for k, v in t.items() if k.startswith('commit_horizon:'))
            mm = ' '.join(f"{k.split(':', 1)[1]}×{v:,}" for k, v in t.items() if k.startswith('miss_mode:'))
            calls = t.get('source:policy', 0)
            empty = '是' if a.get('recipe_off_only_on_empty_library_tasks') else '否'
            out.append(f"| {a['label']} | {t.get('decisions', 0):,} | {t.get('not_segment_head', 0):,} | {ch} | {t.get('acks_over_5_controls', 0):,} | "
                       f"{calls:,} | {t.get('policy_stall_call', 0):,} | {t.get('policy_recipe_off', 0):,} | {t.get('policy_other', 0):,} | "
                       f"{t.get('periodic_gap_set', 0):,} | {mm} | {empty if t.get('policy_recipe_off') else '—'} |")
        out.append('')
    if vla_audit:
        out += ['| 组（π0.5 / GR00T） | 决策行 | policy_tail | 记了 vision 的行 | 模型调用（MISS） | 其中 Q3 停滞门 | 旋钮调用 | 其他 |',
                '|---|---:|---:|---:|---:|---:|---:|---|']
        for a in vla_audit:
            t = a['counts']
            other = ' '.join(f"{k.split(':', 1)[1]}×{v:,}" for k, v in t.items() if k.startswith('call_other:')) or '0'
            out.append(f"| {a['label']} | {t.get('decisions', 0):,} | {t.get('source:policy_tail', 0):,} | {t.get('logged_vision', 0):,} | "
                       f"{t.get('model_calls', 0):,} | {t.get('call_q3_stall', 0):,} | {t.get('call_knob', 0):,} | {other} |")
        out.append('')
    out += ['## 放宽与适配', ''] + [f'{i}. {x}' for i, x in enumerate(result['relaxations'], 1)]
    if result['load_errors']:
        out += ['', '## 读取出错', ''] + [f"- {e['item']}: `{e['error'].strip().splitlines()[-1]}`" for e in result['load_errors']]
    out += ['', '## 读取的批次', '', '| 线 | 批次（实际目录） | 参照 |', '|---|---|---|']
    for line in lines_by.values():
        knob = ', '.join(p['name'] + ('' if p.get('run') is None or Path(p['run']).name == p['name'] else f" → {Path(p['run']).name}")
                         for p in line['parts'] if p['part'] != 'reference' and p.get('name'))
        ref = next((p for p in line['parts'] if p['part'] == 'reference'), {})
        out.append(f"| {line['label']} | {knob} | {ref.get('name')}（{line.get('reference_key') or 'H100'}） |")
    return '\n'.join(out)+'\n'


# --------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', type=int, default=8, help='process pool size (at most 8)')
    p.add_argument('--out', type=Path, default=HERE/'RESULTS_OFF.json')
    p.add_argument('--md', type=Path, default=HERE/'RESULTS_OFF.md')
    p.add_argument('--results-dir', type=Path, default=Path('/nvme/weiland/openpi/r17_lo_1022/results_off'),
                   help='knob.descent.analyze_mt50 per-cell detail (DESCENT.md, descent.json)')
    p.add_argument('--md-only', action='store_true', help='rewrite the markdown from an existing --out JSON')
    args = p.parse_args()
    if args.md_only:
        args.md.write_text(markdown(read(args.out)))
        return
    chains = read(CHAINS) if CHAINS.exists() else {}
    registry = dict(read(ob.REFERENCES)['references'])
    registry.update(chains.get('references', {}))
    r17 = r17_lines(chains, registry)
    vla = vla_lines()
    # One pool for everything a complete line needs, plus complete knob-off runs of pending lines (log check only).
    items = []
    for line in r17.values():
        roots = [j['run'] for j in line['jobs']] + [line['reference']['run']] if line['complete'] else line['knob_off_runs']
        items += [('r17', r) for r in roots if ('r17', r) not in items]
    vla_items = {}
    for key, line in vla.items():
        units = line['units'] if line['complete'] else line['units'][:1] if line['knob_off_complete'] else []
        for u in units:
            for j in u['jobs']:
                vla_items[len(items)] = (key, u, j)
                items.append(('vla_job', j))
        if line['complete']:
            vla_items[len(items)] = (key, None, None)
            items.append(('vla_ref', (line['model'], line['mode'])))
    # Largest per-step logs first, so the pool is not left waiting on one big lane at the end.
    def size(item):
        path = Path(item[1])/'driver/per_step.jsonl' if item[0] == 'r17' else None
        return path.stat().st_size if path is not None and path.exists() else 0
    order = sorted(range(len(items)), key=lambda i: -size(items[i]))
    print(f'loading {len(items)} items with {min(args.jobs, 8)} processes', flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, 8, len(items) or 1))) as pool:
        results = dict(zip(order, pool.map(load, [items[i] for i in order], chunksize=1)))
    loaded = [results[i] for i in range(len(items))]
    errors = []
    for (kind, payload), out in zip(items, loaded):
        if '__error__' in out:
            errors.append(dict(kind=kind, item=payload if kind != 'vla_job' else payload['name'], error=out['__error__']))
            print('ERROR', kind, errors[-1]['item'], out['__error__'].strip().splitlines()[-1], flush=True)
    for (kind, payload), out in zip(items, loaded):
        if kind == 'r17' and '__error__' not in out:
            an._LANES[str(payload)] = out['data']
    on = on_values()
    rows, audits, line_out = [], [], {}
    for key, line in r17.items():
        label = line_label(line)
        status = 'complete' if line['complete'] else 'pending'
        if line['complete'] and any(e['kind'] == 'r17' and e['item'] in [j['run'] for j in line['jobs']] + [line['reference']['run']] for e in errors):
            status = 'error'
        for (kind, payload), out in zip(items, loaded):
            if kind == 'r17' and payload in line['knob_off_runs'] and '__error__' not in out:
                for group, counts in sorted(out['audit'].items()):
                    if group.rsplit('_', 1)[-1] == 'koff':
                        off_tasks = sorted({k.split(':', 1)[1] for k in counts if k.startswith('recipe_off_task:')})
                        audits.append(dict(pipeline='R17', label=f'{label} · {Path(payload).name} · {group}', run=payload, group=group,
                                           counts=counts, recipe_off_tasks=off_tasks,
                                           recipe_off_only_on_empty_library_tasks=set(off_tasks) <= empty_tasks(line['model'])))
        try:
            built = r17_rows(line, args.results_dir) if status == 'complete' else []
        except Exception:
            errors.append(dict(kind='r17_rows', item=label, error=traceback.format_exc()))
            status, built = 'error', []
        if built:
            for r in built:
                s = summary_r17(r)
                rows.append(dict(model=r['model'], benchmark=r['benchmark'], mode=r['mode'], suite=r['suite'], size=r['size'],
                                 label=cell_label(line, r['suite'], r['size']), off=s,
                                 on=on.get((r['model'], r['benchmark'], r['mode'], r['suite'], r['size'])), detail=r))
        line_out[f'{key[0]}:{key[1]}:{key[2]}'] = dict(
            model=key[0], benchmark=key[1], mode=key[2], label=label, status=status, pipeline='R17', gpu=line['gpu'], reference_key=line['reference_key'],
            reference_run=line['reference'] and line['reference']['run'], parts=line['parts'], levels_unit=line['levels_unit'],
            off_decision=str(HERE/'off_decisions'/f"{line['levels_unit']['name']}.json") if line['levels_unit'] else None)
    for key, line in vla.items():
        label = line_label(line)
        mine = {i: v for i, v in vla_items.items() if v[0] == key}
        failed = any('__error__' in loaded[i] for i in mine)
        status = 'complete' if line['complete'] and not failed else 'error' if line['complete'] else 'pending'
        for i, (_, unit, job) in mine.items():
            if unit is not None and unit['stage'] == 'off' and '__error__' not in loaded[i]:
                for arm, counts in sorted(loaded[i]['audit'].items()):
                    audits.append(dict(pipeline=line['pipeline'], label=f'{label} · {arm}', run=job['run'], arm=arm, counts=counts))
        if status == 'complete':
            ref = next(loaded[i] for i, v in mine.items() if v[1] is None)
            line['reference_arms'] = ref['used']
            jobs = [(unit, job, loaded[i]['data']) for i, (_, unit, job) in mine.items() if unit is not None]
            try:
                built = vla_rows(line, jobs, ref['data'])
            except Exception:
                errors.append(dict(kind='vla_rows', item=label, error=traceback.format_exc()))
                status, built = 'error', []
            for r in built:
                s = summary_vla(r)
                rows.append(dict(model=r['model'], benchmark='libero', mode=r['mode'], suite=r['suite'], size=r['size'],
                                 label=cell_label(line, r['suite'], r['size']), off=s,
                                 on=on.get((r['model'], 'libero', r['mode'], r['suite'], r['size'])), detail=r))
        line_out[f'{key[0]}:{key[1]}:{key[2]}'] = dict(
            model=key[0], benchmark=key[1], mode=key[2], label=label, status=status, pipeline=line['pipeline'], gpu='H100', reference_key=None,
            reference_unit=line['reference_unit'], reference_arms=line['reference_arms'], parts=line['parts'],
            off_decision=line['off_decision'])
    for key, line in line_out.items():
        path = line.get('off_decision')
        if line['status'] != 'complete' and path and Path(path).exists():
            line['off_vs_pure_classification'] = dict(Counter(r['status'] for r in read(path)['rows']))
    order_keys = [f'{m}:{b}:{mode}' for b in ('mt50', 'libero') for m in ('groot', 'dp', 'fastwam', 'pi05')
                  for mode in ('reduced', 'full')]
    line_out = {k: line_out[k] for k in sorted(line_out, key=lambda k: order_keys.index(k) if k in order_keys else 99)}
    rank = {k: i for i, k in enumerate(line_out)}
    rows.sort(key=lambda r: (rank[f"{r['model']}:{r['benchmark']}:{r['mode']}"], r['suite'], r['size']))
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), long_segment='off',
                  arm='filtered (success-filtered library)', lines=line_out, rows=rows, decision_log_audit=audits,
                  relaxations=RELAXATIONS, load_errors=errors, final_chains=chains,
                  on_sources=[str(ON_R17), str(ON_VLA)],
                  rule='formal_a_analysis.choose / knob.descent.analyze_mt50 as in RESULTS_1022: untuned iff the complete '
                       'knob-off cohort reaches same-card same-step pure; else per-task first failure, 2-success tolerance',
                  reference='same model / benchmark / steps, same card, C5 pure inference (a model call every 5 controls)',
                  withdrawn_outcomes_used=False, dev_outcomes_used=False)
    dump(args.out, result)
    args.md.write_text(markdown(result))
    print(f'Wrote {args.out} ({len(rows)} cells; ' +
          ', '.join(f"{k}={v['status']}" for k, v in line_out.items()) + f') and {args.md}', flush=True)


if __name__ == '__main__':
    main()
