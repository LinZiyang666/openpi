"""R13 frozen-gate audit. Read B pilot results only; never open test-A results.

Run from repository root:
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src \
 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 .venv/bin/python exp/offline_search/rounds/r13/astra/evaluate_gates.py

Only this new script's gates.json and GATES.md are written. Re-running includes
newly complete pilot arms. Frozen manifest members are checked before/after.
Optional --resume reuses this script's completed arm audits after rechecking
their logged-input/manifest hashes, and audits newly complete arms. Their shard
checks retain the original audit snapshot; omit --resume for a full rehash.
Outcome labels are consumed ONLY by variance_audit(), never by release logic.
"""
from __future__ import annotations

import collections
import ast
import concurrent.futures
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
IMPL = HERE.parent / 'impl'
PILOTS = Path('/home/weiland/trace_runs/os_closed_loop')
SEED = 20261004
BOOT = 5000
CELLS = ['groot_l10_50', 'groot_l10_200', 'groot_spatial_50',
         'pi05_l10_50', 'pi05_l10_200', 'pi05_spatial_50']
COSTS = {'groot': (.148, .852), 'pi05': (.152, .848)}
FAIL_LABEL = 'released-only-as-labelled-Q1-failure measurement'
INPUTS = {}
HASH_CACHE = {}


def sha(path):
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(b)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def record(path):
    path = Path(path)
    # A definitions are permitted; A result paths never are.
    if str(path).startswith(str(PILOTS)):
        assert path.relative_to(PILOTS).parts[0] in {'r13_pilot_' + c for c in CELLS}, path
    INPUTS[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read(path):
    return json.loads(record(path).read_text())


def rows(path):
    with record(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def check_manifest(path, root):
    checks = []
    for line in record(path).read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        p = root / name.lstrip('*')
        checks.append({'path': str(p), 'expected': expected, 'actual': sha(p),
                       'pass': sha(p) == expected})
    return {'count': len(checks), 'pass': all(x['pass'] for x in checks),
            'mismatches': [x for x in checks if not x['pass']]}


def attest_default_off_source(release):
    """Rebuild SHA-bound R13 source in memory and audit later disabled hooks.

    This does not deploy, import R14, or read any R14 results. The current
    shared files contain later optional audit blocks. Prove the entire AST
    outside those disabled blocks is unchanged from the released R13 bytes.
    """
    diff = record(IMPL / 'shared_changes.diff').read_text().splitlines(True)
    results = []
    class WithoutDisabledAudit(ast.NodeTransformer):
        def __init__(self):
            self.removed = 0
        def visit_If(self, node):
            first = node.test.values[0] if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And) else node.test
            disabled = (isinstance(first, ast.Call) and isinstance(first.func, ast.Name)
                        and first.func.id == 'getattr' and len(first.args) == 3
                        and isinstance(first.args[1], ast.Constant) and first.args[1].value == 'os_r14_audit'
                        and isinstance(first.args[2], ast.Constant) and first.args[2].value is False)
            if disabled:
                assert not node.orelse
                self.removed += 1
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            v = node.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == 'add_argument'
                    and v.args and isinstance(v.args[0], ast.Constant) and v.args[0].value == '--os-r14-audit'):
                assert any(k.arg == 'action' and isinstance(k.value, ast.Constant) and k.value.value == 'store_true' for k in v.keywords)
                assert not any(k.arg == 'default' and not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in v.keywords)
                self.removed += 1
                return None
            return self.generic_visit(node)
    for name in ['plugin.py', 'look_recording.py']:
        src = record(IMPL / 'before' / name).read_text().splitlines(True)
        out = []; i = 0; active = False
        for line in diff:
            if line.startswith('--- '):
                active = line.rstrip().endswith('/before/' + name)
                continue
            if not active or line.startswith('+++ '): continue
            if line.startswith('@@'):
                pos = int(re.match(r'@@ -(\d+)', line)[1]) - 1
                out += src[i:pos]; i = pos
            elif line.startswith(' '):
                assert src[i] == line[1:]; out.append(src[i]); i += 1
            elif line.startswith('-'):
                assert src[i] == line[1:]; i += 1
            elif line.startswith('+'):
                out.append(line[1:])
        out += src[i:]; rebuilt = ''.join(out)
        expected = next(x['after_sha256'] for x in release['shared_changes'] if x['path'].endswith('/' + name))
        rebuilt_sha = hashlib.sha256(rebuilt.encode()).hexdigest()
        current = record(REPO / 'exp/offline_search/closed_loop' / name).read_text()
        transform = WithoutDisabledAudit()
        same = ast.dump(transform.visit(ast.parse(current))) == ast.dump(ast.parse(rebuilt))
        results.append({'file': name, 'rebuilt_release_sha256': rebuilt_sha, 'expected_sha256': expected,
                        'disabled_hook_nodes_removed_for_comparison': transform.removed,
                        'default_off_AST_equal': same, 'pass': rebuilt_sha == expected and same})
    return results


def implementation_evidence():
    release = read(IMPL / 'RELEASE.json'); mechanics = read(IMPL / 'mechanics.json')
    mismatches = []
    for name, expected in release['evidence_sha256'].items():
        actual = sha(record(IMPL / name))
        if actual != expected:
            mismatches.append({'file': name, 'expected': expected, 'actual': actual})
    runtime = check_manifest(IMPL / 'H100_SOURCES.sha256', REPO)
    default_off = attest_default_off_source(release)
    suites = ET.parse(record(IMPL / 'mechanics_tests.xml')).getroot().findall('testsuite')
    junit = {k: sum(int(s.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    junit['timestamps'] = [s.get('timestamp') for s in suites]
    # A later mechanics report differs from RELEASE's old digest, but the
    # retained JUnit independently records the full passing test inventory.
    # Do not manufacture a functional failure from a stale summary reference.
    acceptable_evidence_drift = all(x['file'] == 'mechanics.json' for x in mismatches)
    acceptable_source_drift = all(Path(x['path']).name in ['plugin.py', 'look_recording.py'] for x in runtime['mismatches'])
    fits = read(IMPL / 'fits.json')
    fit_checks = [{'config_key': f['config_key'], 'path': f['path'], 'expected': f['sha256'],
                   'actual': sha(f['path']), 'pass': sha(f['path']) == f['sha256']}
                  for f in fits if '_500:' in f['config_key']]
    passed = (release['PASS'] and mechanics['PASS'] and acceptable_evidence_drift
              and acceptable_source_drift and all(x['pass'] for x in default_off)
              and junit['tests'] == release['test_counts']['new']['tests']
              and all(junit[k] == 0 for k in ['failures', 'errors', 'skipped'])
              and all(x['pass'] for x in fit_checks))
    return {'pass': bool(passed), 'release_pass': release['PASS'], 'evidence_mismatches': mismatches,
            'runtime_closure': runtime, 'default_off_source_equivalence': default_off,
            'test_counts': release['test_counts'], 'independent_retained_junit': junit,
            'mechanics': mechanics, 'size500_actual_fit_hash_checks': fit_checks,
            'interpretation': 'RELEASE references a stale mechanics.json digest. Retained JUnit and current report agree on passing mechanics; all non-audit runtime source hashes match, and SHA-reconstructed released R13 source is AST-identical after removing only opt-in default-false R14 audit hooks. No new implementation tests or nonpilot outcome analysis were run.'}


def interval(x):
    return np.quantile(np.asarray(x, float), [.025, .975]).tolist()


def bootstrap_indices(keys):
    rng = np.random.default_rng(SEED)
    tasks = np.array([k[0] for k in keys])
    return np.concatenate([rng.choice(np.flatnonzero(tasks == t), (BOOT, sum(tasks == t)), replace=True)
                           for t in sorted(set(tasks))], axis=1)


def close(a, b):
    return a is not None and math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-11)


def distribution(x):
    x = np.asarray(x, float)
    return {'n': len(x), 'mean': float(x.mean()), 'min': float(x.min()),
            'p25': float(np.quantile(x, .25)), 'p50': float(np.median(x)),
            'p95': float(np.quantile(x, .95)), 'max': float(x.max())} if len(x) else {'n': 0}


def ratio_summary(eps, cv, cm, target):
    N = np.array([e['N'] for e in eps]); V = np.array([e['V'] for e in eps])
    M = np.array([e['M'] for e in eps]); G = np.array([e['G'] for e in eps])
    cost = cv * V + cm * M
    return {'episodes': len(eps), 'N': int(N.sum()), 'V': int(V.sum()), 'M': int(M.sum()),
            'G': int(G.sum()), 'cost': float(cost.sum()),
            'ir': float(cost.sum() / N.sum()), 'error': float(cost.sum() / N.sum() - target),
            'guard_floor_on_observed_states': float((cv * V + cm * G).sum() / N.sum())}


def replay_positions(looks, cfg, cv, cm, mode):
    # Fixed actual look/guard/fallback sequence; alternate M feeds back into theta.
    # Shuffling includes all finite distances within each episode, preserving the
    # nonfinite fallback positions. It never shuffles across identities.
    u = np.array([x['u'] for x in looks], float)
    finite = np.isfinite(u)
    if mode == 'constant':
        u[finite] = .5
    elif mode == 'shuffle':
        rng = np.random.default_rng(SEED)
        u[finite] = rng.permutation(u[finite])
    V = M = 0
    out = []
    for x, score in zip(looks, u):
        theta = cfg['bias'] + (cv * V + cm * M - cfg['target'] * x['step']) / cm / cfg['width']
        call = x['guard'] or x['fallback'] or not np.isfinite(score) or score > theta
        out.append(bool(call and not x['guard'] and finite[len(out)]))
        V += 1
        M += int(call)
    return np.array(out, bool)


def audit_shard(job):
    path, entry, dec, need_shadow = job
    bad = collections.Counter()
    h = sha(path)
    if h != entry['sha256']:
        bad['look_sha_mismatch'] += 1
    # Loading selected tiny arrays does not decompress camera keys/images.
    with np.load(path, allow_pickle=False) as z:
        if int(z['step']) != entry['step'] or int(z['task_id']) != entry['task_id'] or int(z['init']) != entry['init']:
            bad['look_identity_mismatch'] += 1
        raw = float(z['raw_distance'])
        ex = dec.get('extras', {}) if dec else {}
        if 'os_r13_raw_distance' in ex:
            logged = ex['os_r13_raw_distance']
            if math.isfinite(raw) and not close(logged, raw):
                bad['look_raw_distance_mismatch'] += 1
        if need_shadow and dec:
            if 'cache_proposal' not in z or z['cache_proposal'].shape != (10, 32):
                bad['missing_corrected_cache_proposal'] += 1
            # MISS uses its actual policy output; HIT needs a shadow output.
            if dec['hit'] and 'shadow_policy' not in z:
                bad['missing_hit_shadow'] += 1
    return bad, path.stat().st_size


def audit_arm(spec, cfgs, selections, used, fit_by_key):
    cell = spec['cell']; model, suite, size = cell.split('_')
    method = spec.get('method', 'distance_static_b'); rep = spec.get('replicate', 0)
    target = spec['target']; cv, cm = COSTS[model]
    root = PILOTS / ('r13_pilot_' + cell)
    run = root / 'runs' / spec['arm']
    out = {'arm': spec['arm'], 'cell': cell, 'method': method, 'target': target,
           'replicate': rep, 'pilot_root': str(root), 'complete': False,
           'intended_host': 'local_4090' if model == 'groot' else 'h100',
           'actual_host_user_report': 'local_4090' if model == 'groot' or cell == 'pi05_l10_50' else 'h100',
           'host_protocol_pass': cell != 'pi05_l10_50'}
    if not (run / 'summary.json').exists():
        out.update(status='pending', reason='No complete arm summary at audit snapshot.')
        return out, None
    summary = read(run / 'summary.json')
    if summary.get('complete') != spec['episodes']:
        out.update(status='pending', summary_complete=summary.get('complete'), reason='Incomplete arm summary.')
        return out, None
    bad = collections.Counter()
    expected = set(map(tuple, selections['pairs_by_cell'][cell]))
    manifest = read(run / 'manifest.json')
    contracts = read(root / 'dev_contracts.json')
    contract = contracts[spec['arm']]
    def check(ok, name, count=1):
        if not ok:
            bad[name] += count
    check(summary['init_pool'] == manifest['init_pool'] == contract['init_pool'] == 'B', 'not_B')
    check(set(map(tuple, manifest['pairs'])) == expected, 'manifest_pairs')
    # The deployment contract binds load_manifest's selection digest (below),
    # while input_files independently records the complete manifest byte SHA.
    selection_digest = hashlib.sha256(json.dumps(sorted(map(tuple, manifest['pairs'])), separators=(',', ':')).encode()).hexdigest()
    check(selection_digest == contract['manifest_sha256'], 'manifest_selection_digest')
    check(manifest['frozen_selection_sha256'] == sha(HERE / 'validation_selection.json'), 'selection_sha')
    check(manifest['used_identities_sha256'] == sha(HERE / 'used_b_identities.json'), 'used_identity_sha')
    prior = set().union(*(set(map(tuple, v)) for k, v in used.items() if k.endswith('_' + suite)))
    check(not expected & prior, 'used_B_overlap', len(expected & prior))
    excluded = {(int(t), int(i)) for t, ids in manifest['subset']['excluded_inits_by_task'].items() for i in ids}
    check(not expected & excluded, 'served_subset_overlap', len(expected & excluded))
    config_key = spec.get('config_key', f'{cell}:distance_static_b:{target}')
    fit = fit_by_key[config_key]
    check(sha(fit['path']) == fit['sha256'] == contract['fit_sha256'], 'fit_sha')
    cfg = cfgs.get(config_key)
    journal = list(rows(run / 'client/journal.jsonl'))
    accepted = [x for x in journal if x.get('accepted') and x.get('status') in ('done', 'failed') and not x.get('error')]
    check(len(accepted) == len(expected), 'accepted_count')
    amap = {(x['task_uid'], x['attempt']): x for x in accepted}
    check(len(amap) == len(accepted), 'duplicate_accepted_episode')
    observed = {(int(x['task_uid'].split(':')[-2]), int(x['task_uid'].split(':')[-1])) for x in accepted}
    check(observed == expected, 'missing_or_unknown_identity', len(observed ^ expected))
    for x in accepted:
        check(x.get('init_pool') == 'B', 'journal_not_B')
    decs = collections.defaultdict(list); ends = collections.defaultdict(list)
    startups = []; excluded_decs = 0
    for p in sorted(run.glob('server_*/decisions*.jsonl')):
        for x in rows(p):
            if x['ev'] == 'startup':
                startups.append(x)
            elif x['ev'] in ('dec', 'episode'):
                k = (x.get('uid'), x.get('attempt'))
                if k not in amap:
                    excluded_decs += x['ev'] == 'dec'
                elif x['ev'] == 'dec':
                    decs[k].append(x)
                else:
                    ends[k].append(x)
    clients = collections.defaultdict(list)
    excluded_client = 0
    for x in rows(run / 'client/per_step.jsonl'):
        if 'step_idx' not in x:
            continue  # Client episode summary, not a committed decision slot.
        k = (x['task_uid'], x['attempt'])
        if k in amap and x.get('accepted'):
            clients[k].append(x)
        else:
            excluded_client += 1
    host_names = sorted({x['host'] for x in startups})
    seeds = sorted({x.get('rollout_seed', {}).get('seed') for x in startups})
    checkpoints = sorted({x['argv'][x['argv'].index('--checkpoint') + 1] for x in startups if '--checkpoint' in x['argv']})
    for x in startups:
        check('--os-r14-audit' not in x['argv'], 'later_audit_hook_enabled')
        check(x['kwargs']['episode_subset'] == manifest['subset']['episode_ids_by_task'], 'startup_subset')
        check(x.get('seed') == 0, 'knob_coin_seed')
        check(x.get('policy_tail') and x.get('blind') and x.get('policy_tail_blocks') == 1, 'startup_lifecycle')
        check(x.get('kwargs', {}).get('method') == method, 'startup_method')
        check(close(x.get('kwargs', {}).get('target'), target), 'startup_target')
    eps = []; lookup = {}; counters = collections.Counter(); debts = {'startup_step_lt_10': [], 'later_step_ge_10': []}
    theta_all = []; u_all = []; eng = collections.Counter(); const_diff = const_union = shuffle_diff = shuffle_union = 0
    max_cache_look_run = max_cache_slots = max_call_gap = 0
    for k, j in sorted(amap.items(), key=lambda kv: tuple(map(int, kv[0][0].split(':')[-2:]))):
        rr = sorted(decs[k], key=lambda x: x['step']); cc = sorted(clients[k], key=lambda x: x['step_idx'])
        N = len(rr); V = M = G = K = 0; looks = []; run_looks = run_slots = 0; call_steps = []
        check([x['step'] for x in rr] == list(range(N)), 'noncontiguous_or_duplicate_decision')
        check(len(cc) == N and [x['step_idx'] for x in cc] == list(range(0, 5*N, 5)), 'client_server_steps')
        stage1_offset = rr[0]['stage1_calls'] - int(rr[0]['vision']) if rr else 0
        check(len(ends[k]) == 1, 'episode_end_count')
        if ends[k]:
            check(ends[k][0]['n_decisions'] == N, 'episode_N')
            check(ends[k][0]['success'] == j['success'], 'outcome_reconciliation')
        for index, x in enumerate(rr):
            vision = bool(x['vision']); call = not x['hit']; ex = x.get('extras', {})
            lookup[(k[0], k[1], x['step'])] = x
            check(x['ctx_step'] == x['step'], 'context_step')
            check(x.get('ok', True), 'server_not_ok')
            # stage1_calls is connection-lifetime cumulative, unlike R13 V.
            check(x['stage1_calls'] == stage1_offset + V + int(vision), 'stage1_cumulative')
            check((x.get('s23_ms') is not None) == call, 'actual_call_accounting')
            check(not call or vision, 'blind_policy_call')
            if index < len(cc):
                check((cc[index]['hit_type'] == 'FULL_HIT') == x['hit'], 'client_server_hit')
                check(cc[index]['task_id'] == x['task_id'] and cc[index]['orig_init_state_idx'] == x['init'], 'client_server_identity')
                check(cc[index]['run_id'] == j['run_id'], 'accepted_run_id')
            prefix = 'os_r11_' if method == 'periodic' else 'os_r13_'
            guard = knob = False; fallback = False
            if vision:
                has_audit = prefix + 'N' in ex
                check(has_audit, 'missing_knob_audit')
                if has_audit:
                    for key, val in [('N', x['step']), ('V', V), ('M', M)]:
                        check(ex.get(prefix + key) == val, 'ledger_' + key)
                    guard = bool(ex[prefix + 'guard']); knob = bool(ex[prefix + 'knob_call'])
                    check(not (guard and knob), 'double_guard_knob')
                    check(call == (guard or knob), 'guard_knob_OR')
                    reason = int(ex.get('os_reason', 0))
                    expected_reason = {'periodic': 62, 'distance_feedback': 69, 'distance_static_b': 70}[method]
                    check(not knob or reason == expected_reason, 'knob_reason')
                    check(not guard or reason < 61, 'guard_reason_overwritten')
                    if method != 'periodic':
                        d = ex['os_r13_raw_distance']; finite = d is not None and math.isfinite(d) and d >= 0
                        debt = (cv * V + cm * M - target * x['step']) / cm
                        check(close(ex['os_r13_D'], debt), 'debt_arithmetic')
                        check(close(ex['os_r13_target'], target), 'frozen_target')
                        check(bool(ex['os_r13_nonfinite']) == (not finite), 'nonfinite_flag')
                        if method == 'distance_feedback':
                            u = float(np.clip((np.log1p(d) - cfg['lo']) / (cfg['hi'] - cfg['lo']), 0, 1)) if finite else float('nan')
                            theta = cfg['bias'] + debt / cfg['width']
                            for name, val in [('theta', theta), ('width', cfg['width']), ('bias', cfg['bias'])]:
                                check(close(ex['os_r13_' + name], val), 'frozen_' + name)
                            check(not finite or close(ex['os_r13_u'], u), 'score_arithmetic')
                            check(knob == (not guard and (not finite or u > theta)), 'strict_trigger')
                            check(bool(ex['os_r13_saturated_low']) == (theta < 0), 'low_flag')
                            check(bool(ex['os_r13_saturated_high']) == (theta >= 1), 'high_flag')
                            theta_all.append(theta)
                            if finite:
                                u_all.append(u)
                            eng['all_looks'] += 1; eng['nonfinite'] += not finite
                            eng['non_guard_nonfinite'] += not finite and not guard
                            if not guard and finite:
                                eng['eligible'] += 1; eng['interior'] += 0 <= theta < 1
                                eng['theta_low'] += theta < 0; eng['theta_high'] += theta >= 1
                                eng['theta_near_ceiling_ge_0p95'] += theta >= .95
                                eng['score_zero'] += u == 0; eng['score_one'] += u == 1
                            debts['startup_step_lt_10' if x['step'] < 10 else 'later_step_ge_10'].append(debt)
                            looks.append(dict(step=x['step'], guard=guard, fallback=fallback, u=u, call=call))
                        else:
                            check(close(ex['os_r13_theta'], spec['threshold']), 'static_threshold')
                            check(knob == (not guard and (not finite or d > spec['threshold'])), 'static_trigger')
                    else:
                        check(close(ex['os_r11_setting'], cfg['setting']), 'periodic_setting')
                        check(knob == (not guard and ex['os_r11_run'] >= ex['os_r11_cap']), 'periodic_trigger')
                run_looks = 0 if call else run_looks + 1
                max_cache_look_run = max(max_cache_look_run, run_looks)
            else:
                check(prefix + 'N' not in ex, 'knob_on_blind_slot')
            if x['source'] == 'policy_tail':
                check(index > 0 and not rr[index-1]['hit'] and not vision and x['hit'], 'tail_predecessor')
                check(ex.get('policy_anchor_step') == x['step'] - 1, 'tail_anchor')
                if index + 1 < N:
                    check(rr[index+1]['vision'], 'fresh_look_after_tail')
            if call:
                call_steps.append(x['step'])
                if index + 1 < N:
                    if rr[index+1]['source'] != 'policy_tail':
                        counters['tail_fallbacks'] += 1
                        check(rr[index+1]['vision'], 'invalid_tail_fallback')
                else:
                    counters['terminal_call_without_tail'] += 1
            run_slots = 0 if call else run_slots + 1
            max_cache_slots = max(max_cache_slots, run_slots)
            counters['policy_tails'] += x['source'] == 'policy_tail'
            counters['guard_calls'] += guard; counters['knob_calls'] += knob
            counters['vision'] += vision; counters['calls'] += call; counters['decisions'] += 1
            G += int(guard); K += int(knob); V += int(vision); M += int(call)
        if call_steps:
            max_call_gap = max(max_call_gap, max(np.diff([-1] + call_steps + [N]).tolist()))
        if ends[k]:
            check(ends[k][0]['n_miss'] == M, 'episode_M')
        if method == 'distance_feedback':
            actual = np.array([x['call'] and not x['guard'] and np.isfinite(x['u']) for x in looks])
            natural = replay_positions(looks, cfg, cv, cm, 'natural')
            check(np.array_equal(actual, natural), 'natural_replay_positions')
            constant = replay_positions(looks, cfg, cv, cm, 'constant')
            shuffled = replay_positions(looks, cfg, cv, cm, 'shuffle')
            const_diff += int((actual ^ constant).sum()); const_union += int((actual | constant).sum())
            shuffle_diff += int((actual ^ shuffled).sum()); shuffle_union += int((actual | shuffled).sum())
        task, init = map(int, k[0].split(':')[-2:])
        eps.append({'task': task, 'init': init, 'attempt': k[1], 'N': N, 'V': V, 'M': M, 'G': G, 'K': K,
                    'cost': cv * V + cm * M, 'terminal_debt': (cv * V + cm * M - target * N) / cm,
                    'success_for_variance_only': int(j['success'])})
    totals = ratio_summary(eps, cv, cm, target)
    ledger = summary['cost_ledger']
    for name, key in [('N', 'decisions'), ('V', 'vision_decisions'), ('M', 'misses')]:
        check(totals[name] == ledger[key], 'summary_' + name)
    check(totals['M'] == totals['G'] + sum(e['K'] for e in eps), 'terminal_call_partition')
    # Independently bind every accepted recorded look to the exact logged anchor.
    jobs = []; seen = set(); shard_counts = collections.Counter()
    for p in sorted(run.glob('server_*/looks/manifest*.json')):
        lm = read(p)
        check(lm['init_pool'] == lm['contract']['init_pool'] == 'B', 'look_not_B')
        check(lm['contract']['fit_sha256'] == contract['fit_sha256'], 'look_fit_sha')
        for ent in lm['shards']:
            kk = (ent['uid'], ent['attempt'], ent['step'])
            dec = lookup.get(kk)
            if dec is not None:
                check(kk not in seen, 'duplicate_recorded_look')
                seen.add(kk)
                check(dec['vision'], 'recorded_blind_look')
            else:
                shard_counts['excluded_attempt_shards'] += 1
            jobs.append((p.parent / ent['file'], ent, dec, bool(spec.get('shadow'))))
    check(seen == {k for k, x in lookup.items() if x['vision']}, 'missing_or_extra_look',
          len(seen ^ {k for k, x in lookup.items() if x['vision']}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for shard_bad, size_bytes in pool.map(audit_shard, jobs):
            bad.update(shard_bad); shard_counts['shards_hashed_and_checked'] += 1
            shard_counts['bytes_hashed'] += size_bytes
    keys = [(e['task'], e['init']) for e in eps]
    bi = bootstrap_indices(keys)
    costs = np.array([e['cost'] for e in eps]); ns = np.array([e['N'] for e in eps])
    boot_ir = costs[bi].sum(1) / ns[bi].sum(1)
    ir = totals['ir']; budget_pass = abs(ir - target) <= .02
    q25 = float(np.quantile(ns, .25))
    subgroups = {}
    for label, subset in [('short_N_le_p25', [e for e in eps if e['N'] <= q25]),
                          ('longer_N_gt_p25', [e for e in eps if e['N'] > q25]),
                          ('guard_heavy_G_over_V_ge_0p25', [e for e in eps if e['G'] / e['V'] >= .25]),
                          ('guard_light_G_over_V_lt_0p25', [e for e in eps if e['G'] / e['V'] < .25])]:
        if subset:
            subgroups[label] = ratio_summary(subset, cv, cm, target)
    engagement = None
    if method == 'distance_feedback':
        interior = eng['interior'] / eng['eligible'] if eng['eligible'] else 0.
        change = const_diff / const_union if const_union else 0.
        q = (2 * target - cv) / cm
        engagement = {**dict(eng), 'threshold_interior_fraction': interior,
                      'constant_symmetric_difference': const_diff, 'constant_union': const_union,
                      'constant_selected_position_change': change,
                      'shuffle_symmetric_difference': shuffle_diff, 'shuffle_union': shuffle_union,
                      'shuffle_selected_position_change': shuffle_diff / shuffle_union if shuffle_union else 0.,
                      'pass_interior': interior >= .40, 'pass_constant': change >= .20,
                      'pass': interior >= .40 and change >= .20,
                      'theta': distribution(theta_all), 'score': distribution(u_all),
                      'debt_by_phase': {k: distribution(v) for k, v in debts.items()},
                      'conditional_no_guard_debt_bound': [min(0., -cfg['width'] * cfg['bias'] - q),
                                                        max(0., cfg['width'] * (1 - cfg['bias']) + 1 - q)],
                      'conditional_bound_assumptions': 'No guards, finite scores, nominal two-slot cadence, reset at zero; not a universal cache-gap cap.'}
    out.update(complete=True, status='evaluated', totals=totals, ir_ci95=interval(boot_ir),
               signed_error_ci95=interval(boot_ir-target), budget_pass=budget_pass,
               engagement=engagement, mechanics={'pass': not bad, 'violations': dict(bad), 'counts': dict(counters),
               'excluded_server_decisions': excluded_decs, 'excluded_client_rows': excluded_client,
               'journal_attempts': len(journal), 'accepted_episodes': len(accepted),
               'incomplete_or_unaccepted_attempts': [{k: x.get(k) for k in ['task_uid','attempt','status','accepted','init_pool']}
                                                    for x in journal if x not in accepted], **dict(shard_counts)},
               observed_server_hosts=host_names, checkpoints=checkpoints, rollout_seeds=seeds,
               fit_sha256=fit['sha256'], fit_path=fit['path'],
               summary_alternative_cost_not_gate={k: ledger.get(k) for k in ['ir_per_request', 'ir_measured_per_five_controls']},
               operating_checks={'episode_N': distribution(ns), 'short_episode_cut_N': q25,
                                 'subgroups': subgroups, 'terminal_debt': distribution([e['terminal_debt'] for e in eps]),
                                 'max_cache_only_real_look_run': max_cache_look_run,
                                 'max_no_call_decision_slots': max_cache_slots,
                                 'max_no_call_nominal_controls': 5 * max_cache_slots,
                                 'max_call_gap_slots_including_boundaries': max_call_gap},
               episodes=eps)
    return out, {'keys': keys, 'cost': costs, 'N': ns, 'boot_ir': boot_ir,
                 'outcomes': np.array([e['success_for_variance_only'] for e in eps])}


def paired_metrics(arms, arrays):
    pairs = []
    for a in arms:
        if a['method'] != 'distance_feedback':
            continue
        p = next(x for x in arms if x['cell'] == a['cell'] and x['method'] == 'periodic'
                 and x['target'] == a['target'] and x['replicate'] == a['replicate'])
        row = {'cell': a['cell'], 'target': a['target'], 'replicate': a['replicate'],
               'distance_arm': a['arm'], 'periodic_arm': p['arm'], 'complete': a['complete'] and p['complete']}
        if row['complete']:
            aa, pp = arrays[a['arm']], arrays[p['arm']]
            assert aa['keys'] == pp['keys']
            d = a['totals']['ir'] - p['totals']['ir']
            row.update(distance_minus_periodic_ir=d, absolute_ir_difference=abs(d),
                       ci95=interval(aa['boot_ir'] - pp['boot_ir']), matched_budget_pass=abs(d) <= .01)
        pairs.append(row)
    return pairs


def within_task_cov(x, keys):
    # Fixed tasks: sample covariance within tasks, then equal task average.
    task = np.array([k[0] for k in keys])
    return np.mean([np.cov(x[task == t], rowvar=False, ddof=1) for t in sorted(set(task))], axis=0)


def variance_audit(arms, arrays):
    zsum = statistics.NormalDist().inv_cdf(.975) + statistics.NormalDist().inv_cdf(.8)
    details = []; by_cell = {}; keys_cell = {}
    for cell in CELLS:
        if not cell.endswith('_50'):
            continue
        aa = [next((a for a in arms if a['cell'] == cell and a['method'] == m and a['target'] == .32 and a['replicate'] == r and a['complete']), None)
              for r in [0, 1] for m in ['distance_feedback', 'periodic']]
        if any(a is None for a in aa):
            details.append({'cell': cell, 'status': 'pending_duplicate_arms'})
            continue
        ar = [arrays[a['arm']] for a in aa]
        assert all(a['keys'] == ar[0]['keys'] for a in ar)
        x = np.column_stack([ar[0]['outcomes'] - ar[1]['outcomes'], ar[2]['outcomes'] - ar[3]['outcomes']])
        keys = ar[0]['keys']; cov = within_task_cov(x, keys); v = float(np.trace(cov) / 2); c = float(cov[0, 1])
        rho = c / math.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else None
        # Nonnegative exchangeable components are a planning sensitivity, not an
        # identified variance decomposition from only two repeats.
        floor = max(0., min(v, c)); noise = max(0., v - floor)
        bi = bootstrap_indices(keys); bx = x[bi]
        centered = bx.reshape(BOOT, 10, 10, 2)
        centered = centered - centered.mean(axis=2, keepdims=True)
        bcov = np.einsum('btni,btnj->bij', centered, centered) / (10 * 9)
        brho = np.divide(bcov[:,0,1], np.sqrt(bcov[:,0,0]*bcov[:,1,1]), out=np.zeros(BOOT), where=bcov[:,0,0]*bcov[:,1,1] > 0)
        details.append({'cell': cell, 'status': 'evaluated', 'identities': len(keys),
                        'paired_discordance_by_replicate': np.mean(x*x, axis=0).tolist(),
                        'within_task_covariance': cov.tolist(), 'replicate_correlation': rho,
                        'replicate_correlation_bootstrap_ci95': interval(brho),
                        'replicate_paired_difference_agreement': float(np.mean(x[:,0] == x[:,1])),
                        'distance_outcome_repeat_agreement': float(np.mean(ar[0]['outcomes'] == ar[2]['outcomes'])),
                        'periodic_outcome_repeat_agreement': float(np.mean(ar[1]['outcomes'] == ar[3]['outcomes'])),
                        'variance_component_floor_clipped': floor, 'variance_component_noise_clipped': noise,
                        'four_replicate_cell_MDE80_pp_at_500_inits': 100*zsum*math.sqrt((floor+noise/4)/500),
                        'infinite_replicate_cell_MDE80_floor_pp_at_500_inits': 100*zsum*math.sqrt(floor/500),
                        'rollout_seeds': [a['rollout_seeds'] for a in aa]})
        by_cell[cell] = x; keys_cell[cell] = keys
    complete_suites = []
    for suite in ['l10', 'spatial']:
        cells = [f'{m}_{suite}_50' for m in ['groot', 'pi05']]
        if all(c in by_cell for c in cells):
            assert keys_cell[cells[0]] == keys_cell[cells[1]]
            x = (by_cell[cells[0]] + by_cell[cells[1]]) / 2
            cov = within_task_cov(x, keys_cell[cells[0]])
            v = float(np.trace(cov)/2); c = float(cov[0,1]); floor = max(0., min(v,c))
            complete_suites.append({'suite': suite, 'averaged_models': cells, 'v': v, 'covariance': c,
                                    'floor': floor, 'noise': max(0., v-floor)})
    pooled = {'status': 'not_identified_missing_duplicate_cell', 'complete_suites': complete_suites}
    if len(complete_suites) == 2:
        def se(rep):
            return math.sqrt(sum(s['floor'] + (s['noise']/rep if math.isfinite(rep) else 0.) for s in complete_suites) / (4*500))
        pooled.update(status='planning_scenario_only', four_replicate_MDE80_pp=100*zsum*se(4),
                      infinite_replicate_MDE80_floor_pp=100*zsum*se(float('inf')),
                      independent_replicate_MDE80_pp=100*zsum*math.sqrt(sum(s['v'] for s in complete_suites)/(4*500*4)),
                      perfect_dependence_MDE80_pp=100*zsum*math.sqrt(sum(s['v'] for s in complete_suites)/(4*500)))
    return {'per_cell': details, 'pooled': pooled,
            'use': 'Variance/power only. No SR ranking, method selection, release gate, or dose selection.',
            'limitations': 'Two repeats estimate dependence imprecisely. Size-50/.32 does not identify covariance across unreplicated sizes or .40. Full-grid precision is not established. Process seeds are not independent per-episode streams; shadows use private noise but rep0 shadow overhead can alter concurrent request order. Missing model/suite cells are not reweighted away.',
            'planning_assumptions': '500 A inits per task-balanced cell; four exchangeable repeats; normal two-sided alpha=.05, power=.80; no independent-sample credit for repeated sizes/doses; clipped components shown alongside raw covariance. Within-task covariance conditions on benchmark tasks.'}


def release_cells(arms, pairs, impl_pass):
    cells = []
    for cell in CELLS:
        aa = [a for a in arms if a['cell'] == cell and a['method'] != 'distance_static_b']
        pp = [p for p in pairs if p['cell'] == cell]
        completed = all(a['complete'] for a in aa)
        bad_budget = [a['arm'] for a in aa if a['complete'] and not a['budget_pass']]
        bad_eng = [a['arm'] for a in aa if a['complete'] and a['engagement'] and not a['engagement']['pass']]
        bad_pair = [f"{p['target']}:rep{p['replicate']}" for p in pp if p['complete'] and not p['matched_budget_pass']]
        mechanics = impl_pass and completed and all(a.get('mechanics', {}).get('pass', False) for a in aa)
        host = all(a['host_protocol_pass'] for a in aa)
        reasons = []
        if not completed: reasons.append('pending primary pilot arms')
        if completed and not mechanics: reasons.append('mechanics or provenance gate failed')
        if not host: reasons.append('pi05 L10-50 B pilot on local 4090; frozen A host H100; same-host pilot condition unmet')
        if bad_budget: reasons.append('primary arm aggregate budget gate failed')
        if bad_eng: reasons.append('distance engagement gate failed')
        if bad_pair: reasons.append('direct matched-budget label fails at listed dose/replicate; SPEC does not make this alone a budget/engagement stop')
        if not completed or not mechanics or not host:
            verdict = 'held'
        elif bad_budget or bad_eng:
            verdict = FAIL_LABEL
        else:
            verdict = 'released'
        cells.append({'cell': cell, 'verdict': verdict, 'validated_target_control': verdict == 'released',
                      'all_doses_direct_matched_budget': completed and not bad_pair,
                      'all_primary_arms_complete': completed, 'mechanics_pass': mechanics, 'same_host_pass': host,
                      'budget_failing_arms': bad_budget, 'engagement_failing_arms': bad_eng,
                      'unmatched_dose_replicates': bad_pair, 'reasons': reasons,
                      'scope': 'Frozen measurement only; Q1 failure is held from validated-control/matched-frontier claims; no tuning.'})
    for model in ['groot', 'pi05']:
        # SPEC says "available smaller-library cells", without a model-specific
        # exception. The shared-controller transfer release awaits all six.
        deps = [c for c in cells if not c['cell'].endswith('_500')]
        ok = impl_pass and all(c['verdict'] == 'released' for c in deps)
        cells.append({'cell': model+'_l10_500', 'verdict': 'released' if ok else 'held',
                      'validated_target_control': False, 'label': 'library-only transfer; never B-validated',
                      'mechanics_pass': impl_pass, 'prerequisite_cells': [c['cell'] for c in deps],
                      'reasons': [] if ok else ['SPEC requires the same controller to pass available smaller-library cells (all six; no model-specific exception is stated). Pending/host-mismatched pi05 L10-50 leaves this transfer prerequisite unmet. Q1-failure exception does not waive it.']})
    return cells


def release_roots(cells):
    roots = read(IMPL / 'test_a_roots.json')
    queue = record(IMPL / 'queue_test_a.txt').read_text().splitlines()
    cellmap = {c['cell']: c for c in cells}; result = []
    assert len(queue) == len(roots)
    for index, (root, line) in enumerate(zip(roots, queue)):
        left, armtext = line.split('|'); host, path = left.split(maxsplit=1)
        assert path == root['final_root'] and armtext.split() == root['arms']
        definitions = {name: sha(record(Path(root['root']) / name)) == digest
                       for name, digest in root['definitions_sha256'].items()}
        assert all(definitions.values()), ('staged definition changed', path)
        staged_arms = read(Path(root['root']) / 'arms.json')
        assert all('--os-r14-audit' not in a.get('plugin_args', []) for a in staged_arms)
        dependencies = sorted({m.group(1) for arm in root['arms'] if (m := re.match(r'r13_((?:groot|pi05)_(?:l10|spatial)_(?:50|200|500))_', arm))})
        statuses = [cellmap[c]['verdict'] for c in dependencies]
        verdict = 'held' if 'held' in statuses else FAIL_LABEL if FAIL_LABEL in statuses else 'released'
        result.append({'root': root['final_root'], 'frozen_queue_index': index, 'host': root['intended_host'],
                       'replicate': int(re.search(r'_rep(\d+)$', path)[1]), 'arm_count': len(root['arms']),
                       'cells': dependencies, 'verdict': verdict,
                       'frozen_definition_hash_checks': definitions,
                       'reason': 'Whole frozen root includes a held cell; no arm dropping/reordering authorized.' if verdict == 'held' else
                                 'Only clearly labelled Q1 failure measurement; not a passed matched-budget comparison.' if verdict == FAIL_LABEL else 'All cell gates satisfied.',
                       'execution_condition': 'Preserve frozen order, hosts and complete replicate blocks; later blocks wait for both fleets to complete/collect prior block. This audit launches nothing.'})
    for i, row in enumerate(result):
        row['held_predecessor_roots_in_frozen_execution_order'] = [prev['root'] for prev in result[:i]
                if prev['verdict'] == 'held' and (prev['host'] == row['host'] or prev['replicate'] < row['replicate'])]
    return result


def fmt(x, digits=4):
    return '—' if x is None else f'{x:.{digits}f}'


def render(result):
    arms = result['arms']; cells = result['cells']; roots = result['test_a_roots']
    cm = {c['cell']: c for c in cells}
    complete = [a for a in arms if a['complete']]
    lines = ['# R13 B pilot 放行审计', '',
             'GR00T 结论：' + '；'.join(f"{c.removeprefix('groot_')}：{cm[c]['verdict']}" for c in ['groot_l10_50','groot_l10_200','groot_spatial_50','groot_l10_500']) + '。', '',
             'GR00T Spatial 已通过内容放行门槛；L10 冻结整批含 size-500，transfer 前提未满足时整批 held。冻结本地队列先 L10 后 Spatial，因此不能跳过 held 的 L10、删 arm 或改顺序来声称现在整批可启动。后续 replicate block 仍须等上一完整 block 的两个 fleet 均完成。', '',
             '规则解释：SPEC 把配对 IR 差 ≤.01 写为直接同预算标签的条件；停止/重设计段落明确针对预算或参与度失败。因此配对差单独失败会取消该点的同预算标签，不冒充控制器预算失败。Size-500 的前提写的是全部可用小库 cell，没有按模型豁免；目前 pi05 L10-50 尚未完成同机验证，两种模型的 size-500 均 held。', '',
             f"本次纳入 {len(complete)} 个完整 B arm、{sum(a['totals']['episodes'] for a in complete)} 条 episode；其余 {len(arms)-len(complete)} 个 arm 待完成。pi05 L10-50 运行主机改为本地 4090（用户说明是容量决定、未看结果）；即便数据补齐，也不等于冻结要求的 H100 同机验证。脚本可重跑补齐。", '',
             '判定只用预算、距离参与度、机械及协议检查；成功标签仅进入重复方差/功效审计，没有按 SR 选方法、剂量或 cell。预算或参与度失败保留为 Q1 failure，交回另行版本化的 B-only 重设计；不在本验证集或 A 上调参。', '',
             f"审计时间：{result['timestamp_utc']}。冻结文件校验 {result['frozen_before']['count']} 项，前后均 {'通过' if result['frozen_before']['pass'] and result['frozen_after']['pass'] else '失败'}。", '',
             '| Cell | Verdict | Budget failing arms | Engagement failing arms | Unmatched pairs |',
             '| --- | --- | ---: | ---: | ---: |']
    order = ['groot_l10_50','groot_l10_200','groot_spatial_50','groot_l10_500','pi05_l10_50','pi05_l10_200','pi05_spatial_50','pi05_l10_500']
    for c in order:
        row = cm[c]
        lines.append(f"| {c} | {row['verdict']} | {len(row.get('budget_failing_arms', []))} | {len(row.get('engagement_failing_arms', []))} | {len(row.get('unmatched_dose_replicates', []))} |")
    complexity = result['calibration_complexity']
    lines += ['', 'Technical conventions: frozen owner cost = .148 V + .852 M (GR00T), .152 V + .848 M (pi05); IR = pooled cost / pooled N. Do not substitute summary latency or current cost-table IR. 95% percentile intervals use 5,000 paired episode/init-cluster resamples, seed 20261004, within each fixed task. Point estimates, not interval endpoints, determine the frozen ≤.02 and |D−P|≤.01 gates. Repeats are checked separately; no pooling rescues failures.', '',
              f"Calibration complexity across the frozen grid: feedback {complexity['feedback_total_independent_scalars']} scalars (two scale parameters/cell, one bias/cell/dose, one selected shared width); periodic {complexity['periodic_total_scalar_caps']} library-solver caps; static {complexity['static_total_scalar_thresholds']} B-only diagnostic thresholds. Exact B/library source paths are retained per cell in gates.json. No scalar is refit here.", '',
              '| Cell | Method / target / rep | N / V / M | IR [95% CI] | Error | Budget |',
              '| --- | --- | --- | --- | ---: | --- |']
    for a in arms:
        label = {'distance_feedback':'D','periodic':'P','distance_static_b':'S'}[a['method']]
        if not a['complete']:
            lines.append(f"| {a['cell']} | {label} / {a['target']:.2f} / {a['replicate']} | — | pending | — | pending |")
            continue
        t = a['totals']; lo, hi = a['ir_ci95']
        lines.append(f"| {a['cell']} | {label} / {a['target']:.2f} / {a['replicate']} | {t['N']}/{t['V']}/{t['M']} | {t['ir']:.4f} [{lo:.4f}, {hi:.4f}] | {t['error']:+.4f} | {'pass' if a['budget_pass'] else 'FAIL'} |")
    lines += ['', 'D = distance_feedback; P = periodic; S = static-b diagnostic. Static failures remain calibration failures and do not select a replacement. Full per-arm counts, guard floors, short/long and guard-heavy operating checks, startup/later/terminal debt, raw-score saturation, hosts and hashes are in gates.json.', '',
              '| Cell | Target / rep | D−P IR [95% paired CI] | Matched |',
              '| --- | --- | --- | --- |']
    for p in result['pairs']:
        if not p['complete']: continue
        lo, hi = p['ci95']
        lines.append(f"| {p['cell']} | {p['target']:.2f} / {p['replicate']} | {p['distance_minus_periodic_ir']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {'pass' if p['matched_budget_pass'] else 'FAIL'} |")
    lines += ['', 'Engagement: denominators exclude guards and nonfinite fallback looks. Constant replay substitutes u=.5 and recomputes its own call ledger on the same actual look/guard sequence; change = symmetric difference / union of eligible selected positions. Shuffling is within episode (seed 20261004), descriptive only. These are fixed-state diagnostics, not counterfactual rollouts. “Near ceiling” is explicitly descriptive theta≥.95; high saturation is theta≥1.', '',
              '| Cell | Target / rep | Eligible | Interior ≥.40 | Constant ≥.20 | Shuffle | theta <0 / ≥1 / ≥.95 | Nonfinite | Gate |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for a in complete:
        e = a['engagement']
        if e:
            lines.append(f"| {a['cell']} | {a['target']:.2f} / {a['replicate']} | {e['eligible']} | {e['threshold_interior_fraction']:.3f} | {e['constant_selected_position_change']:.3f} | {e['shuffle_selected_position_change']:.3f} | {e['theta_low']}/{e['theta_high']}/{e['theta_near_ceiling_ge_0p95']} | {e['nonfinite']} | {'pass' if e['pass'] else 'FAIL'} |")
    lines += ['', '| Cell | Complete arms | Logged decisions | Mechanics violations | Guard / knob calls | Max no-call slots |',
              '| --- | ---: | ---: | ---: | --- | ---: |']
    for c in CELLS:
        aa = [a for a in complete if a['cell'] == c]
        if not aa: continue
        lines.append(f"| {c} | {len(aa)} | {sum(a['totals']['N'] for a in aa)} | {sum(sum(a['mechanics']['violations'].values()) for a in aa)} | {sum(a['mechanics']['counts']['guard_calls'] for a in aa)}/{sum(a['mechanics']['counts']['knob_calls'] for a in aa)} | {max(a['operating_checks']['max_no_call_decision_slots'] for a in aa)} |")
    imp = result['implementation_evidence']
    lines += ['', f"Implementation evidence: functional pass={imp['pass']}. The old RELEASE digest is stale for {len(imp['evidence_mismatches'])} evidence file (mechanics.json); retained JUnit independently records {imp['independent_retained_junit']['tests']} passing tests. Current runtime differs in {len(imp['runtime_closure']['mismatches'])} shared files only through default-off audit hooks: the audit reconstructs the SHA-bound released files from saved originals + shared_changes.diff and verifies AST equality with those disabled hooks removed. No R14 result is read. The full drift evidence is retained in gates.json.", '',
              f"Live checks cover accepted N/V/M prefixes, OR/reason arithmetic, one actual MISS charge, client/server/terminal reconciliation, tail transitions, identities, fit bindings, and {sum(a['mechanics']['shards_hashed_and_checked'] for a in complete)} look shard SHA/raw-distance checks. Existing CPU evidence supplies off equivalence, retry/reset/clone/order isolation and invalid-tail/early-end cases; live logs cannot directly count hidden internal guard invocations. All {len(imp['size500_actual_fit_hash_checks'])} size-500 fit hashes are checked against implementation records. Remote loaded code/checkpoint bytes cannot be freshly attested by this local audit; logged paths/hosts/seeds and fit/contract hashes are retained.", '',
              '| Duplicate .32 cell | Paired discordance r0 / r1 | Within-task variance r0 / r1 | Covariance | rho [95% CI] | 4-rep cell MDE / infinite-rep floor (pp) |',
              '| --- | --- | --- | ---: | --- | --- |']
    for p in result['power']['per_cell']:
        if p['status'] != 'evaluated':
            lines.append(f"| {p['cell']} | pending | — | — | — | — |")
            continue
        v = p['within_task_covariance']; d = p['paired_discordance_by_replicate']; ci = p['replicate_correlation_bootstrap_ci95']
        lines.append(f"| {p['cell']} | {d[0]:.3f}/{d[1]:.3f} | {v[0][0]:.4f}/{v[1][1]:.4f} | {v[0][1]:.4f} | {fmt(p['replicate_correlation'],3)} [{ci[0]:.3f}, {ci[1]:.3f}] | {p['four_replicate_cell_MDE80_pp_at_500_inits']:.2f}/{p['infinite_replicate_cell_MDE80_floor_pp_at_500_inits']:.2f} |")
    lines += ['', 'Power audit: ' + result['power']['pooled']['status'] + '. ' + result['power']['limitations'], '',
              'MDE values are scenarios at 500 inits/cell, alpha=.05 two-sided and 80% power. Raw task-conditional covariance is reported; the exchangeable floor is clipped to [0, mean variance] only for projections. Pilot mean SR differences are not used or ranked. Two .32 repeats cannot certify pooled .32/.40, all-size precision; no precision-based arm selection or extra independent-episode credit is taken.', '',
              '| Frozen test-A batch root | Host | Verdict |', '| --- | --- | --- |']
    for r in sorted(roots, key=lambda x: ('groot' not in x['root'], x['frozen_queue_index'])):
        lines.append(f"| {Path(r['root']).name} | {r['host']} | {r['verdict']} |")
    lines += ['', 'Root verdict is the content-gate decision for the complete frozen batch, including controls. Scheduling is separate: the frozen queue places each L10 root before Spatial on its fleet. Released Spatial roots cannot bypass held L10 predecessors without changing that plan; blocked predecessors are enumerated in gates.json. Rep1–rep3 also wait for prior complete blocks on both fleets. No installation, rollout, remote command or GPU work was performed.', '',
              'Reproduce (from repository root; only the three new deliverables are in astra):', '', '```bash',
              'taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r13/astra/evaluate_gates.py', '```', '']
    return '\n'.join(lines)


def main():
    allowed = set(range(10,22)) | set(range(54,66))
    assert os.sched_getaffinity(0) <= allowed, 'Use the requested taskset CPU affinity.'
    before = check_manifest(HERE / 'MANIFEST.sha256', HERE)
    assert before['pass'], before
    frozen = {line.split(maxsplit=1)[1].lstrip('*') for line in (HERE / 'MANIFEST.sha256').read_text().splitlines()}
    assert not {'evaluate_gates.py', 'GATES.md', 'gates.json'} & frozen
    primary = read(HERE / 'pilot_grid.json'); static = read(HERE / 'static_pilot_grid.json')
    specs = sorted(primary + static, key=lambda a: (CELLS.index(a['cell']), a.get('method','distance_static_b'), a['target'], a.get('replicate',0)))
    cfgs = {x['config_key']: x for x in read(HERE / 'configs.json')}
    for c in read(HERE / 'periodic_fits.json'):
        cfgs[f"{c['cell']}:periodic:{c['target']}"] = c
    selection = read(HERE / 'validation_selection.json'); used = read(HERE / 'used_b_identities.json')
    fit_by_key = {x['config_key']: x for x in read(IMPL / 'fits.json')}
    implementation = implementation_evidence()
    impl_pass = implementation['pass']
    previous = None; cached = {}; resume_source = None
    if '--resume' in sys.argv:
        previous = json.loads((HERE / 'gates.json').read_text())
        assert previous['schema'] == 'r13_frozen_B_pilot_gates_v1'
        resume_source = {'sha256': sha(HERE / 'gates.json'), 'timestamp_utc': previous['timestamp_utc'],
                         'note': 'Complete arm numerical audits reused; logged inputs and shard manifests rehashed; shard content hashes retain their original full-audit snapshot.'}
        for path, meta in previous['input_files'].items():
            if path.startswith(str(PILOTS)):
                assert sha(path) == meta['sha256'], ('cached input changed', path)
        INPUTS.update({p: m for p, m in previous['input_files'].items() if p not in INPUTS})
        cached = {a['arm']: a for a in previous['arms'] if a['complete']}
    arms = []; arrays = {}
    for spec in specs:
        print('AUDIT', spec['arm'], flush=True)
        if spec['arm'] in cached:
            a = cached[spec['arm']]
            check_fit = fit_by_key[spec.get('config_key', f"{spec['cell']}:distance_static_b:{spec['target']}")]
            assert a['fit_sha256'] == sha(check_fit['path'])
            eps = a['episodes']; keys = [(e['task'],e['init']) for e in eps]
            cost = np.array([e['cost'] for e in eps]); ns = np.array([e['N'] for e in eps])
            bi = bootstrap_indices(keys)
            ar = {'keys':keys, 'cost':cost, 'N':ns, 'boot_ir':cost[bi].sum(1)/ns[bi].sum(1),
                  'outcomes':np.array([e['success_for_variance_only'] for e in eps])}
            assert np.allclose(interval(ar['boot_ir']), a['ir_ci95'], rtol=0, atol=1e-14)
            a.setdefault('full_audit_snapshot_utc', previous['timestamp_utc'])
        else:
            a, ar = audit_arm(spec, cfgs, selection, used, fit_by_key)
        arms.append(a)
        if ar is not None:
            arrays[a['arm']] = ar
            print('RESULT', a['arm'], 'IR', round(a['totals']['ir'],6), 'budget', a['budget_pass'],
                  'engagement', a['engagement']['pass'] if a['engagement'] else None,
                  'violations', a['mechanics']['violations'], flush=True)
    pairs = paired_metrics(arms, arrays)
    cells = release_cells(arms, pairs, impl_pass)
    roots = release_roots(cells)
    power = variance_audit(arms, arrays)
    for a in arms:
        p = next((p for p in pairs if a['arm'] in (p['distance_arm'],p['periodic_arm'])), None)
        a['direct_matched_budget_label'] = p.get('matched_budget_pass') if p and p['complete'] else None
        if not a['complete'] or not a['host_protocol_pass'] or not a['mechanics']['pass'] or not impl_pass:
            a['verdict'] = 'held'
        elif a['method'] == 'distance_static_b':
            a['verdict'] = 'diagnostic-cost-pass' if a['budget_pass'] else 'diagnostic-calibration-failure'
        elif not a['budget_pass'] or (a['engagement'] and not a['engagement']['pass']):
            a['verdict'] = FAIL_LABEL
        else:
            a['verdict'] = 'released'
    frozen_cfgs = read(HERE / 'configs.json')
    complexity = {'feedback_cell_scale_scalars': 2, 'feedback_bias_scalars_per_dose': 1,
                  'feedback_shared_selected_widths': len({c['width'] for c in frozen_cfgs if c['method']=='distance_feedback'}),
                  'feedback_total_independent_scalars': 2*len({c['cell'] for c in frozen_cfgs if c['method']=='distance_feedback'}) + len([c for c in frozen_cfgs if c['method']=='distance_feedback']) + 1,
                  'periodic_total_scalar_caps': len(read(HERE/'periodic_fits.json')),
                  'static_total_scalar_thresholds': len(static),
                  'feedback_calibration_sources_by_cell': {c['cell']:c['sources'] for c in frozen_cfgs if c['method']=='distance_feedback'},
                  'periodic_source': 'unchanged R11 whole-episode-held-out library ScheduleReplay solver',
                  'static_source': 'same B calibration paths as feedback; frozen minimax thresholds; never retuned here'}
    after = check_manifest(HERE / 'MANIFEST.sha256', HERE)
    assert after['pass'], after
    result = {'schema': 'r13_frozen_B_pilot_gates_v1', 'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'data_boundary': 'Only six explicit B pilot roots; A definitions read only for release mapping, no A results opened.',
              'bootstrap': {'resamples': BOOT, 'seed': SEED, 'cluster': 'original task/init episode identity', 'stratification': 'suite/task', 'interval': '95% percentile', 'cost': 'ratio of pooled cost and N'},
              'frozen_before': before, 'frozen_after': after,
              'implementation_evidence': implementation,
              'release_rule_interpretation': {'pairing': 'SPEC says ≤.01 is required for a direct matched-budget label; its stop/redesign paragraph names budget or engagement failure. A paired-only mismatch is flagged, not relabelled as a target-control failure.',
                                              'size500': 'Require all available smaller-library cells, as written; SPEC gives no model-specific exemption. Thus both transfer cells await all six valid same-host pilots.'},
              'resume_source': resume_source, 'calibration_complexity': complexity,
              'arms': arms, 'pairs': pairs, 'cells': cells, 'test_a_roots': roots, 'power': power,
              'input_files': INPUTS}
    # Replace only this audit's newly introduced outputs; no frozen writes.
    for name, content in [('gates.json', json.dumps(result, indent=2, allow_nan=False)+'\n'), ('GATES.md', render(result))]:
        (HERE / name).write_text(content)
    print('VERDICTS', json.dumps(cells, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
