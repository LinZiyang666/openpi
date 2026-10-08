"""Read-only confirmation of frozen B tables; writes only beside this script.

Run with the CPU command in HANDBACK.md. No fitting, selection, launch, GPU,
or mutation of B files. Frozen fit relocation uses the existing CPU-only
hash verifier. A is measurement only.
"""
from __future__ import annotations

import ast
import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys
import subprocess

import numpy as np

from .analyze import (Reader, Episodes, ServedFitVerifier, require, sha, verify_protocol as _verify_protocol, paired,
                     uid_pair, markdown_table, cell_name, write_outputs)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
EMITTER = HERE / 'test_a_selected'
from .prepare import CELLS
ARMS = ('pure_c10', 'uniform', 'task_descent')
SEED, DRAWS = 20261005, 5000
EXPECTED = [(t, i) for t in range(10) for i in range(50)]
UNCHANGED_FUNCTIONS = ('sum_sample', 'ci', 'paired_boot', 'ir_savings',
                       'pool_stats', 'optimism', '_calculate_group', 'calculate',
                       'f', 'interval_text')


def verify_protocol(reader):
    """Verify the registered stage adapter, retaining actual-byte input audits."""
    source = Path(__file__).resolve()
    current = reader.read(source)
    original = reader.read(HERE / 'test_a_stage_original_analyze_test_a.py.txt')
    adapter = reader.json(HERE / 'test_a_stage_adapter.json')
    protocol_bytes = reader.read(HERE / 'protocol.json')
    source_name = str(source.relative_to(REPO))
    require(adapter['schema'] == 'r15.descent.test_a_stage_adapter.v1' and
            adapter['source'] == source_name and adapter['protocol_sha256'] == sha(protocol_bytes),
            'Wrong test-A stage adapter registration')
    require(adapter['frozen_sha256'] == json.loads(protocol_bytes)['input_sha256'][source_name] and
            sha(original) == adapter['frozen_sha256'] and sha(current) == adapter['adapted_sha256'],
            'Unregistered test-A source change')
    def functions(data):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(data).body
                if isinstance(n, ast.FunctionDef)}
    before, after = functions(original), functions(current)
    for name in UNCHANGED_FUNCTIONS:
        require(before[name] == after[name] and
                sha(before[name].encode()) == adapter['unchanged_function_ast_sha256'][name],
                f'Frozen test-A measurement/loader function changed: {name}')
    fix = adapter['served_fit_loader_fix']
    baseline = reader.read(HERE / 'analyze_test_a.before_served_fit_fix.py.txt')
    require(sha(baseline) == fix['before_source_sha256'] and
            functions(baseline)['load_a'] == before['load_a'] and
            sha(before['load_a'].encode()) == adapter['unchanged_function_ast_sha256']['load_a'] and
            sha(after['load_a'].encode()) == fix['load_a_ast_sha256'],
            'Unregistered test-A served-fit loader change')
    require(fix['hash_verifier_source'] == 'exp/offline_search/rounds/r15/descent/analyze.py' and
            sha(reader.read(REPO / fix['hash_verifier_source'])) == fix['hash_verifier_source_sha256'] and
            fix['relocation_source'] == 'exp/offline_search/closed_loop/ops/h100/assets.py' and
            sha(reader.read(REPO / fix['relocation_source'])) == fix['relocation_source_sha256'],
            'Test-A served-fit verification implementation changed')

    class FrozenSourceView:
        # Only this already-validated source is projected to its frozen bytes.
        # The underlying Reader still records and rehashes the current file.
        def read(self, path):
            data = reader.read(path)
            return original if Path(path).resolve() == source else data

        def json(self, path):
            return json.loads(self.read(path))

    return _verify_protocol(FrozenSourceView())


def stage_cells(stage, levels, uniforms, reader):
    """Accept a complete model stage or the legacy complete twelve-cell stage."""
    stage = Path(stage).resolve()
    require(stage.is_relative_to(HERE.resolve()), 'Stage must remain under R15 descent')
    staged = reader.json(stage / 'input.json')
    require(set(staged) == {'levels', 'uniform'} and
            isinstance(staged['levels'], dict) and isinstance(staged['uniform'], dict),
            'Invalid emitter input')
    cells = set(staged['levels'])
    require(cells == set(staged['uniform']), 'Stage level/uniform cells differ')
    model_sets = [set(c for c in CELLS if c.split('_')[0] == model) for model in ('pi05', 'groot')]
    require(cells == set(CELLS) or any(cells == model_cells for model_cells in model_sets),
            'Stage must contain one complete model cell set or all registered cells')
    selected = tuple(c for c in CELLS if c in cells)
    expected = dict(levels={c: levels[c] for c in selected}, uniform={c: uniforms[c] for c in selected})
    require(staged == expected, 'Emitter input differs from B for stage cells')
    return selected


def stage_scope(stages, levels, uniforms, reader, combined=False):
    require(len(stages) == (2 if combined else 1), 'Use two model stages for combine; one stage for apply')
    stage_sets = [stage_cells(stage, levels, uniforms, reader) for stage in stages]
    if combined:
        require(all(len(cells) == 6 for cells in stage_sets) and
                set(stage_sets[0]).isdisjoint(stage_sets[1]) and
                set(stage_sets[0]) | set(stage_sets[1]) == set(CELLS),
                'Combine requires disjoint complete pi05 and groot stages')
    cells = tuple(c for c in CELLS if any(c in subset for subset in stage_sets))
    return dict(cells=list(cells), models=sorted({c.split('_')[0] for c in cells}),
                partial=set(cells) != set(CELLS), combined=combined,
                stages=[str(Path(stage).resolve()) for stage in stages])


def calculate_scope(all_arms, b, p, levels, uniforms, scope):
    """Scope the unchanged size-group measurement; preserve its frozen family."""
    global CELLS
    complete = CELLS
    selected = tuple(scope['cells'])
    try:
        CELLS = selected
        result = calculate(all_arms, b, p, {c: levels[c] for c in selected},
                           {c: uniforms[c] for c in selected})
    finally:
        CELLS = complete
    result['analysis_scope'] = scope
    if scope['partial']:
        for group in result['groups_by_library_size'].values():
            group['partial'] = True
            group['methods']['pooled_sr'] = (
                'partial single-model x suite x task mixture, 1000 episode outcomes per arm; '
                '1000 original suite/task/init clusters; all-cell pooling deferred to combine')
    return result


def output_names(scope):
    suffix = '_' + scope['models'][0] if scope['partial'] else ''
    return {name: Path(name).stem + suffix + Path(name).suffix for name in
            ('test_a.json', 'TEST_A.md', 'REPORT.md', 'ANALYSIS_TEST_A_HANDBACK.md')}


def stage_freezes(scope, full_provenance, levels, uniforms, reader):
    """Bind each model stage to the provenance that actually froze its tables."""
    full_time = datetime.fromisoformat(full_provenance['analyzed_at_utc']).timestamp()
    if not (scope['partial'] or scope['combined']):
        # Preserve the original chronology source for a legacy full stage.
        return {'full': dict(source=str(HERE / 'provenance.json'),
                             analyzed_at_utc=full_provenance['analyzed_at_utc'],
                             timestamp=full_time, equivalence_required=False)}
    freezes = {}
    for model in scope['models']:
        path = HERE / f'provenance_{model}.json'
        require(path.is_file(), f'Missing model freeze provenance: {model}')
        partial = reader.json(path)
        cells = [c for c in CELLS if c.split('_')[0] == model]
        partial_scope = partial['analysis_scope']
        require(partial['outcome_pool'] == 'B' and partial_scope['partial'] is True and
                partial_scope['models'] == [model] and partial_scope['cells'] == cells,
                f'Wrong model freeze provenance: {model}')
        table_hashes = {}
        for kind, table in (('levels', levels), ('uniform', uniforms)):
            name = f'{kind}_{model}.json'
            data = reader.read(HERE / name)
            require(sha(data) == partial['selected_output_sha256'][name],
                    f'Frozen partial B output changed: {name}')
            require(json.loads(data) == {c: table[c] for c in cells},
                    f'Partial freeze table differs from full B: {name}')
            table_hashes[name] = sha(data)
        timestamp = datetime.fromisoformat(partial['analyzed_at_utc']).timestamp()
        equivalence_required = full_time > timestamp
        if equivalence_required:
            eq_path = HERE / 'partial_equivalence.json'
            require(eq_path.is_file(), f'Missing partial/full equivalence: {model}')
            eq_bytes = reader.read(eq_path)
            require(full_provenance['selected_output_sha256'].get(eq_path.name) == sha(eq_bytes),
                    f'Partial/full equivalence is not bound to full B provenance: {model}')
            equivalence = json.loads(eq_bytes)
            require(equivalence.get('PASS') is True and model in equivalence.get('checked_models', []),
                    f'Partial/full equivalence did not PASS for model: {model}')
            for name, digest in table_hashes.items():
                checks = [c for c in equivalence.get('checks', []) if c.get('model') == model and c.get('file') == name]
                require(len(checks) == 1 and checks[0].get('byte_identical') is True and
                        checks[0].get('sha256') == digest,
                        f'Missing/mismatched partial/full equivalence check: {name}')
        freezes[model] = dict(source=str(path), source_sha256=sha(reader.read(path)),
            analyzed_at_utc=partial['analyzed_at_utc'], timestamp=timestamp,
            frozen_partial_table_sha256=table_hashes, equivalence_required=equivalence_required,
            equivalence_sha256=sha(eq_bytes) if equivalence_required else None)
    return freezes


def check_freeze_order(all_arms, freezes):
    """Require each stage's own freeze to precede every new-arm completion."""
    evidence = {}
    for model, freeze in freezes.items():
        terminals = [ep.audit['earliest_accepted_terminal_ts']
                     for cell, arms in all_arms.items() if model == 'full' or cell.split('_')[0] == model
                     for label, ep in arms.items() if label != 'pure_c10']
        require(all(freeze['timestamp'] < ts for ts in terminals),
                f'B analysis not earlier than A completion records: {model}')
        first = min(terminals)
        evidence[model] = dict(freeze, earliest_new_arm_completion_ts=first, strict_freeze_before_test=True)
    return evidence


class AServedFitVerifier(ServedFitVerifier):
    """Bind startup paths and decision hashes to one exact frozen fit."""
    def __init__(self, root, row, source_sha256, reader):
        from exp.offline_search.closed_loop.ops.h100.assets import flag
        super().__init__(root, row, source_sha256, reader)
        self.fit = Path(flag(row['plugin_args'], '--os-fit-artifact'))
        require(sha(reader.read(self.fit)) == source_sha256, 'Fit bytes changed')
        self.relocated_path = None
        self.observed_paths = {}
        self.active_path_branch = None
        self.relocation_arm_order = None

    def relocated_hash(self):
        from exp.offline_search.closed_loop.ops.h100.assets import flag
        # The inherited branch verifies the exact source, sealed H100 host,
        # actual remote root and source store before any batch replay.
        expected = super().relocated_hash()
        plan = self.reader.json(self.root / 'h100_sync/plan.json')
        order = [r['arm'] for r in plan.get('arms', [])]
        require(len(order) == len(set(order)), 'Duplicate H100 sync arms')
        if len(order) <= 1:
            self.relocation_arm_order = [self.row['arm']]
            return expected
        frozen = {r['arm']: r for r in self.reader.json(self.root / 'arms.json')}
        require(self.row['arm'] in order and set(order) <= set(frozen), 'Unsealed H100 sync arm order')
        prefix = order[:order.index(self.row['arm']) + 1]
        self.relocation_arm_order = prefix
        if len(prefix) == 1:
            return expected
        seal = self.reader.json(self.root / 'ROOT.json')
        store = Path(flag(self.row['plugin_args'], '--os-root'))
        sources = []
        for name in prefix:
            row = frozen[name]
            require(row.get('intended_host') == 'h100' and Path(flag(row['plugin_args'], '--os-root')) == store,
                    'H100 batch source settings differ')
            fit = flag(row['plugin_args'], '--os-fit-artifact')
            source = self.reader.read(fit)
            require(sha(source) == seal['fit_sha256'][fit], 'H100 batch source fit changed')
            sources.append(source)
        self.relocated_sha256 = h100_relocated_batch_sha256(sources, self.root, store,
                                                           Path(self.remote_run).parent.parent)
        return self.relocated_sha256

    def check_path(self, observed):
        from exp.offline_search.closed_loop.ops.h100.assets import flag, remap
        path = Path(observed)
        if path == self.fit:
            branch = 'frozen_source'
        else:
            if self.relocated_sha256 is None:
                self.relocated_hash()
            expected = Path(remap(str(self.fit), self.root,
                                 Path(flag(self.row['plugin_args'], '--os-root')),
                                 Path(self.remote_run).parent.parent))
            require(path == expected, 'Wrong served artifact: neither frozen source nor deterministic H100 path')
            self.relocated_path = str(expected)
            branch = 'h100_relocated'
        self.observed_paths[str(path)] = branch
        self.active_path_branch = branch
        return branch

    def check(self, observed):
        branch = super().check(observed)
        if self.active_path_branch is not None:
            expected = self.source_sha256 if self.active_path_branch == 'frozen_source' else self.relocated_sha256
            require(observed == expected, 'Served fit path/hash branch mismatch')
            # An unchanged pickle can have identical source/remote hashes.
            branch = self.active_path_branch
            self.observed[observed] = branch
        return branch

    def audit(self):
        return dict(super().audit(), source_path=str(self.fit), relocated_path=self.relocated_path,
                    observed_path_branches=dict(sorted(self.observed_paths.items())),
                    relocation_arm_order=self.relocation_arm_order)


def h100_relocated_batch_sha256(sources, root, store, base):
    """Match assets.build_plan's ordered unpickle/relocate/pickle CPU process."""
    code = '''import hashlib, pickle, sys
from pathlib import Path
from exp.offline_search.closed_loop.ops.h100.assets import relocated_fit
root, store, base = (Path(p) for p in sys.argv[1:])
while True:
    header = sys.stdin.buffer.read(8)
    if not header:
        break
    size = int.from_bytes(header, 'big')
    source = sys.stdin.buffer.read(size)
    assert len(header) == 8 and len(source) == size
    blob = pickle.loads(source)
    blob, changed = relocated_fit(blob, root, store, base)
    content = pickle.dumps(blob, protocol=4) if changed else source
    digest = hashlib.sha256(content).hexdigest()
print(digest)
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(REPO) + ':' + str(REPO / 'src'))
    data = b''.join(len(source).to_bytes(8, 'big') + source for source in sources)
    result = subprocess.run(['taskset', '-c', '2-9,46-53', sys.executable, '-c', code,
                             str(root), str(store), str(base)], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=True)
    digest = result.stdout.decode().strip()
    require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid H100 batch hash result')
    return digest


def load_a(cell, levels, uniform, p, caps, reader, template):
    target=TRACE/('r15_descent_test_'+cell)
    target_seal=reader.json(target/'ROOT.json')
    require(target_seal['cell']==cell and target_seal['init_pool']=='A' and
            target_seal['arm_count']==2 and target_seal['episodes_per_arm']==500, 'Wrong R15 A design')
    table=reader.json(target/'descent_table.json')
    require(table['levels']==levels and table['uniform']==uniform, 'Changed frozen selection')
    require(table['template_sha256']==sha(template), 'Wrong A template hash')
    require(table['off_tasks']==[t for t in range(10) if levels[str(t)]=='off'], 'Wrong off mask')
    capvec=[caps['cells'][cell]['periodic_cap_by_level'][levels[str(t)]] for t in range(10)]
    require(table['caps']==capvec, 'Wrong frozen caps')
    for level,digest in table['source_artifacts'].items():
        require(digest==caps['cells'][cell]['test_fit_sha256_by_level'][level], 'Wrong cap source')
    shared=p['test_references'][cell.rsplit('_',1)[0]]
    require(table['shared_c10_reference']==shared, 'Changed shared test C10 reference')
    arms={};positions={pair:j for j,pair in enumerate(EXPECTED)}
    for arm in ARMS:
        root=Path(shared['root']) if arm=='pure_c10' else target
        seal=reader.json(root/'ROOT.json')
        require(seal['init_pool']=='A' and seal['episodes_per_arm']==500, 'Wrong A reference pool')
        if arm=='pure_c10':
            require(sha(reader.read(root/'arms.json'))==shared['arms_sha256'], 'Changed R14 C10 reference definition')
        for filename,digest in seal['definitions_sha256'].items():
            path=(root/filename).resolve()
            require(path.is_relative_to(root.resolve()) and sha(reader.read(path))==digest, 'Changed A definition')
        require(reader.read(root/'eval500.json')==template, 'Changed A identities')
        rows=reader.json(root/'arms.json')
        if arm!='pure_c10':
            names={f'r15dt_{cell}_uniform_{uniform}',f'r15dt_{cell}_task_descent'}
            require(len(rows)==2 and {r['arm'] for r in rows}==names, 'Wrong R15 A arm set')
        name=shared['arm'] if arm=='pure_c10' else f'r15dt_{cell}_'+('uniform_'+uniform if arm=='uniform' else arm)
        row=next(r for r in rows if r['arm']==name)
        run = root / 'runs' / name
        level = 'c10' if arm == 'pure_c10' else uniform if arm == 'uniform' else 'task_descent'
        require(row['init_pool'] == 'A' and row['dev'] is False and row['episodes'] == 500 and
                row['r14_descent_level'] == level and row['model'] == cell.split('_')[0], 'Wrong arm metadata')
        require(row['replan_steps'] == 5 and row['full_model'] and not row['recording'], 'Wrong execution settings')
        require(Path(row['manifest']) == root / 'eval500.json', 'Wrong arm manifest pointer')
        require(reader.read(run / 'manifest.json') == template, 'Changed per-arm manifest')
        selected = reader.json(run / 'selection.json')
        # Legacy A selection.json only records the manifest pointer. A identity
        # is independently explicit in the sealed arm, every journal and summary.
        require(selected.get('init_pool', 'A') == 'A' and Path(selected['manifest']) == run / 'manifest.json', 'Wrong selection binding')
        fit = Path(row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1])
        fit_digest = seal['fit_sha256'][str(fit)]
        if arm == 'task_descent':
            require(row['kwargs']['task_levels'] == levels and row['kwargs']['task_caps'] == capvec,
                    'Changed adapter settings')
            require(table['fit']['sha256'] == fit_digest and Path(table['fit']['path']) == fit, 'Wrong adapter fit')
        else:
            require(fit_digest == (caps['cells'][cell]['test_fit_sha256_by_level'][level] if arm != 'pure_c10' else shared['fit_sha256']), 'Wrong frozen reference fit')
        fit_verifier = AServedFitVerifier(root, row, fit_digest, reader)

        accepted, errors, journal_count, duplicate_accepted = {}, 0, 0, 0
        for r in reader.jsonl(run / 'client/journal.jsonl'):
            journal_count += 1
            require(r.get('init_pool') == 'A', 'Non-A journal row')
            uid = r.get('task_uid')
            if uid:
                require(uid.startswith(name + ':') and uid_pair(uid) in positions, 'Unexpected journal UID')
            errors += bool(r.get('error'))
            if r.get('accepted') and r.get('status') in ('done', 'failed') and not r.get('error'):
                require(type(r.get('success')) is bool, 'Nonbinary journal outcome')
                duplicate_accepted += uid in accepted
                accepted[uid] = r
        by_pair = {uid_pair(uid): r for uid, r in accepted.items()}
        require(len(accepted) == len(by_pair) == 500 and set(by_pair) == set(EXPECTED), 'Missing/extra A episodes')
        y = np.array([by_pair[x]['success'] for x in EXPECTED], dtype=np.int64).reshape(10, 50)
        summary = reader.json(run / 'summary.json')
        require(summary['init_pool'] == 'A' and summary['arm'] == name and summary['model'] == row['model'], 'Wrong summary')
        require(summary['complete'] == 500 and summary['success'] == int(y.sum()) and
                math.isclose(summary['sr'], float(y.mean()), abs_tol=1e-12), 'Summary SR mismatch')
        require(summary['journal_rows'] == journal_count and summary['error_rows'] == errors, 'Summary journal mismatch')
        retried = sum(int(r.get('attempt', 1) or 1) > 1 for r in accepted.values())
        require(summary['retried'] == retried, 'Summary retry mismatch')
        require(all(summary['per_task'][str(t)] == {'n': 50, 'success': int(y[t].sum())} for t in range(10)),
                'Task summary mismatch')

        decs, duplicates, rejected, hosts, audit_rows = {}, 0, 0, set(), 0
        shards = sorted(run.glob('server_*/decisions_*.jsonl'))
        require(shards, 'Missing decision shards')
        reader.shards[str(run)] = list(map(str, shards))
        for path in shards:
            for d in reader.jsonl(path):
                if d.get('ev') == 'startup':
                    hosts.add(d['host'])
                    require(d['kwargs'] == row['kwargs'] and d['method_spec'] == row['method'], 'Wrong served method')
                    args = d['argv']
                    fit_verifier.check_path(args[args.index('--os-fit-artifact') + 1])
                    continue
                if d.get('ev') != 'dec':
                    continue
                uid = d['uid']
                require(uid.startswith(name + ':') and uid_pair(uid) in positions, 'Unexpected decision UID')
                require(d.get('init_pool', 'A') == 'A', 'Wrong decision pool')
                if uid not in accepted or int(d.get('attempt', 1) or 1) != int(accepted[uid].get('attempt', 1) or 1):
                    rejected += 1
                    continue
                step = d['step']
                require(type(step) is int and step >= 0, 'Invalid decision step')
                require(type(d.get('hit')) is bool and type(d.get('vision')) is bool, 'Nonboolean decision cost')
                require(d.get('ok', True) and (d['hit'] or d['vision']), 'Invalid full-policy decision')
                require(d.get('stage1_mode', 'full') == 'full' and d.get('replan_steps', 5) == 5, 'Unsupported cost mode')
                require(d['hit'] or float(d['miss_k']) == float(summary['cost_ledger']['full_stage_costs']['k']), 'Partial miss')
                require(int(d['task_id']) == uid_pair(uid)[0] and int(d['init']) == uid_pair(uid)[1], 'Wrong decision identity')
                diag = d.get('r14')
                if diag:
                    audit_rows += 1
                    fit_verifier.check(diag['fit_sha256'])
                    if arm == 'task_descent' and 'descent_level' in diag:
                        require(diag['descent_level'] == levels[str(uid_pair(uid)[0])], 'Served task level mismatch')
                key = (uid, step)
                duplicates += key in decs
                # Keep only the last accepted-attempt decision in sorted shard/line order.
                decs[key] = (d['vision'], not d['hit'])
        n, v, m = (np.zeros(500, dtype=np.int64) for _ in range(3))
        steps = {uid: [] for uid in accepted}
        for (uid, step), (vision, miss) in decs.items():
            j = positions[uid_pair(uid)]
            n[j] += 1; v[j] += vision; m[j] += miss
            steps[uid].append(step)
        require(all(s and sorted(s) == list(range(len(s))) for s in steps.values()), 'Noncontiguous decisions')
        ledger = summary['cost_ledger']; full = ledger['full_stage_costs']
        require([int(x.sum()) for x in (n, v, m)] == [ledger[k] for k in ('decisions', 'vision_decisions', 'misses')],
                'N/V/M ledger mismatch')
        # k is the model's full denoising step count (10 for pi05, 8 for GR00T),
        # not the action chunk length. Each miss was checked against it above.
        require(ledger['controls'] == int(n.sum()) * 5 and float(full['k']) > 0., 'Wrong controls or policy length')
        require(set(ledger['l_per_request']['counts']) == {'5.0'} and
                set(ledger['stage1_modes']) == {'full'}, 'Ledger cost-mode mismatch')
        stage_ir = (full['s1'] * v.sum() + (full['s2'] + full['s3']) * m.sum()) / (
            sum(full[k] for k in ('s1', 's2', 's3')) * n.sum())
        require(math.isclose(stage_ir, ledger['ir_per_five_controls'], abs_tol=1e-8), 'Stage ledger IR mismatch')
        cv, cm = p['cost']['coefficients'][row['model']]
        arms[arm] = Episodes(y, n.reshape(10, 50), v.reshape(10, 50), m.reshape(10, 50),
                            (cv * v + cm * m).reshape(10, 50), dict(
            arm=name, root=str(root), accepted_episodes=500, journal_rows=journal_count, error_rows=errors,
            duplicate_accepted_rows=duplicate_accepted, duplicate_decisions=duplicates,
            rejected_attempt_decisions=rejected, retried_episodes=retried,
            selection_pool_field_present='init_pool' in selected,
            earliest_accepted_terminal_ts=min(r['ts'] for r in accepted.values()),
            hosts=sorted(hosts), decision_shards=len(shards), r14_audit_rows=audit_rows,
            summary_stage_ir=stage_ir, full_stage_costs=full, frozen_fit_sha256=fit_digest,
            fit_hash_verification=fit_verifier.audit()))
    return arms


def sum_sample(a, idx):
    return a[np.arange(10)[None, :, None], idx].sum(axis=(1, 2))


def ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def paired_boot(y, ref, sample_delta):
    gains = int(((y == 1) & (ref == 0)).sum())
    losses = int(((y == 0) & (ref == 1)).sum())
    return dict(n=int(y.size), gains=gains, losses=losses, delta=(gains-losses)/y.size,
                bootstrap_ci95=ci(sample_delta), bootstrap_lower95=float(np.quantile(sample_delta, .05)))


def ir_savings(task_stats, ref_stats, task_boot, ref_boot):
    delta = ref_stats['ir'] - task_stats['ir']
    bs = ref_boot['ir'] - task_boot['ir']
    return dict(ir_saved=delta, relative_ir_saved=delta/ref_stats['ir'], bootstrap_ci95=ci(bs),
                cost_saved_per_episode=ref_stats['cost_per_episode']-task_stats['cost_per_episode'],
                relative_cost_saved_per_episode=1-task_stats['cost_per_episode']/ref_stats['cost_per_episode'],
                relative_bootstrap_ci95=ci(bs/ref_boot['ir']))


def pool_stats(stats, coefficients):
    n = sum(s['decisions'] for s in stats)
    cost = sum(cv * s['vision'] + cm * s['misses'] for s, (cv, cm) in zip(stats, coefficients, strict=True))
    episodes = sum(s['n_episodes'] for s in stats)
    successes = sum(s['successes'] for s in stats)
    return dict(n_episodes=episodes, successes=successes, sr=successes/episodes, decisions=n,
                vision=sum(s['vision'] for s in stats), misses=sum(s['misses'] for s in stats),
                normalized_cost=cost, ir=cost/n, cost_per_episode=cost/episodes)


def optimism(bstats, astats):
    out = {'arms': {}}
    for arm in ARMS:
        b, a = bstats[arm], astats[arm]
        out['arms'][arm] = dict(b_predicted=b, test_a_measured=a,
                               sr_b_minus_a=b['sr']-a['sr'], ir_a_minus_b=a['ir']-b['ir'])
    out['contrasts'] = {}
    for arm, ref in [('task_descent', 'pure_c10'), ('uniform', 'pure_c10'), ('task_descent', 'uniform')]:
        bd = bstats[arm]['sr'] - bstats[ref]['sr']; ad = astats[arm]['sr'] - astats[ref]['sr']
        bs = bstats[ref]['ir'] - bstats[arm]['ir']; ass = astats[ref]['ir'] - astats[arm]['ir']
        out['contrasts'][arm + '_vs_' + ref] = dict(b_sr_delta=bd, a_sr_delta=ad,
            sr_advantage_optimism=bd-ad, b_ir_saved=bs, a_ir_saved=ass, ir_saving_optimism=bs-ass)
    return out


def _calculate_group(all_arms, b, p, levels, uniforms):
    # Same sampled original suite/task/init for every arm AND both models.
    rng = np.random.default_rng(SEED)
    idxs = {suite: rng.integers(0, 50, size=(DRAWS, 10, 50)) for suite in ('l10', 'spatial')}
    result = dict(measurement_only=True, primary_tolerance_pp=5, levels=levels, uniform=uniforms,
        methods=dict(seed=SEED, replicates=DRAWS, resampling='whole paired episode within fixed suite/task; shared across arms and models',
            bootstrap_interval='percentile 2.5/97.5%; one-sided lower 5%; numpy linear quantiles',
            task_exact_lower='CP_low(g,n,.025)-CP_high(l,n,.025); no identical-policy shortcut for reruns',
            task_simultaneous_lower='CP tails .05/(2*40), forty model-task comparisons per contrast family',
            pooled_sr='equal model x suite x task mixture, 2000 episode outcomes per arm; 1000 original suite/task/init clusters',
            pooled_ir='sum_cell(cV_cell*V_cell+cM_cell*M_cell)/sum_cell(N_cell), not average cell IR',
            coefficients=p['cost']['coefficients'], inference_scope='fixed tasks and observed single runs; no between-run uncertainty'),
        cells={}, pooled={})
    boots, bstats = {}, {}
    for cell in CELLS:
        arms = all_arms[cell]; idx = idxs[cell.split('_')[1]]
        stats = {a: arms[a].stats() for a in ARMS}
        boot = {}
        for a in ARMS:
            ep = arms[a]
            bs = {k: sum_sample(getattr(ep, k), idx) for k in ('y', 'n', 'cost')}
            bs['sr'] = bs['y']/500; bs['ir'] = bs['cost']/bs['n']; boot[a] = bs
            stats[a]['normalized_cost'] = float(ep.cost.sum())
        c = dict(arms=stats, comparisons={}, ir_savings={}, tasks={}, audit={a: arms[a].audit for a in ARMS})
        for a, ref in [('task_descent', 'pure_c10'), ('uniform', 'pure_c10'), ('task_descent', 'uniform')]:
            c['comparisons'][a + '_vs_' + ref] = paired_boot(arms[a].y, arms[ref].y, boot[a]['sr']-boot[ref]['sr'])
        for ref in ('pure_c10', 'uniform'):
            c['ir_savings']['vs_' + ref] = ir_savings(stats['task_descent'], stats[ref], boot['task_descent'], boot[ref])
        for t in range(10):
            tr = dict(level=levels[cell][str(t)], pure_successes=int(arms['pure_c10'].y[t].sum()),
                      task_successes=int(arms['task_descent'].y[t].sum()), uniform_successes=int(arms['uniform'].y[t].sum()), n=50)
            for a in ('task_descent', 'uniform'):
                exact = paired(arms[a].y[t], arms['pure_c10'].y[t], family=40)
                delta = (arms[a].y[t] - arms['pure_c10'].y[t])[idx[:, t, :]].mean(axis=1)
                exact.update(bootstrap_ci95=ci(delta), bootstrap_lower95=float(np.quantile(delta, .05)),
                             below_minus_5pp=100*(exact['successes']-exact['pure_successes']) < -5*50,
                             exact_lower_within_5pp=exact['lower95'] >= -.05,
                             simultaneous_lower_within_5pp=exact['simultaneous_lower95'] >= -.05)
                tr[a + '_vs_pure'] = exact
            c['tasks'][str(t)] = tr
        c['task_tolerance'] = {a: dict(
            below_minus_5pp=[t for t, tr in c['tasks'].items() if tr[a+'_vs_pure']['below_minus_5pp']],
            exact_lower_within_5pp=[t for t, tr in c['tasks'].items() if tr[a+'_vs_pure']['exact_lower_within_5pp']],
            simultaneous_lower_within_5pp=[t for t, tr in c['tasks'].items() if tr[a+'_vs_pure']['simultaneous_lower_within_5pp']])
            for a in ('task_descent', 'uniform')}
        bcell = b['cells'][cell]; tol = bcell['tolerances']['5']
        require(all(arms[a].audit['full_stage_costs'] == bcell['arm_stats']['c10']['full_stage_costs'] for a in ARMS),
                'A full-policy cost specification differs from frozen B')
        bstats[cell] = dict(pure_c10=bcell['arm_stats']['c10'], uniform=tol['savings']['uniform'], task_descent=tol['savings']['task'])
        c['selection_optimism'] = optimism(bstats[cell], stats)
        c['frozen_b_library_agreement'] = tol['library']
        c['frozen_b_matched_ir'] = bcell['matched_ir']
        result['cells'][cell] = c; boots[cell] = boot
    coeffs = [p['cost']['coefficients'][c.split('_')[0]] for c in CELLS]
    pool = result['pooled']; pool['arms'] = {}; pool['comparisons'] = {}; pool['ir_savings'] = {}
    pb, bpool = {}, {}
    for arm in ARMS:
        pool['arms'][arm] = pool_stats([result['cells'][c]['arms'][arm] for c in CELLS], coeffs)
        bpool[arm] = pool_stats([bstats[c][arm] for c in CELLS], coeffs)
        samples = {k: sum(boots[c][arm][k] for c in CELLS) for k in ('y', 'n', 'cost')}
        samples['sr'] = samples['y']/(500*len(CELLS)); samples['ir'] = samples['cost']/samples['n']; pb[arm] = samples
    for a, ref in [('task_descent', 'pure_c10'), ('uniform', 'pure_c10'), ('task_descent', 'uniform')]:
        pool['comparisons'][a + '_vs_' + ref] = paired_boot(np.stack([all_arms[c][a].y for c in CELLS]),
            np.stack([all_arms[c][ref].y for c in CELLS]), pb[a]['sr'] - pb[ref]['sr'])
    for ref in ('pure_c10', 'uniform'):
        pool['ir_savings']['vs_' + ref] = ir_savings(pool['arms']['task_descent'], pool['arms'][ref], pb['task_descent'], pb[ref])
    pool['task_tolerance'] = {a: {key: [f'{c}:{t}' for c in CELLS for t in result['cells'][c]['task_tolerance'][a][key]]
        for key in ('below_minus_5pp', 'exact_lower_within_5pp', 'simultaneous_lower_within_5pp')}
        for a in ('task_descent', 'uniform')}
    pool['selection_optimism'] = optimism(bpool, pool['arms'])
    return result


def f(value, scale=1, digits=2):
    return f'{value*scale:.{digits}f}'


def interval_text(value, scale=100):
    return '[' + ', '.join(f(x, scale) for x in value) + ']'


def calculate(all_arms, b, p, levels, uniforms):
    global CELLS
    complete=CELLS
    groups={}
    try:
        for size in (100,200,500):
            CELLS=tuple(c for c in complete if c.endswith('_'+str(size)))
            groups[str(size)]=_calculate_group(all_arms,b,p,levels,uniforms)
    finally:
        CELLS=complete
    result=dict(groups_by_library_size=groups,measurement_only=True,
                shared_c10=True,pooled_reference_deduplicated_by_size=True)
    result['cells']={c:v for group in groups.values() for c,v in group['cells'].items()}
    # Existing four-cell report math is reused separately for every library size.
    result['pooled']=groups['100']['pooled']
    result['levels']=levels;result['uniform']=uniforms
    return result


def scopes(r):
    if 'groups_by_library_size' in r:
        for size,group in r['groups_by_library_size'].items():
            yield 'pooled_'+size,group['pooled']
    else:
        yield 'pooled',r['pooled']
    yield from r['cells'].items()


def stats_table(result):
    return markdown_table(['Cell', 'Arm', 'Success / episodes', 'SR %', 'N', 'V', 'M', 'Owner IR', 'Cost / episode'], [
        [cell, arm, f"{s['successes']}/{s['n_episodes']}", f(s['sr'],100), s['decisions'], s['vision'], s['misses'], f(s['ir'],1,6), f(s['cost_per_episode'],1,6)]
        for cell, c in scopes(result) for arm, s in c['arms'].items()])


def _render_test_group(r):
    scope_text = (
        'Partial measurement: this library-size group uses the two suite cells of one complete model. '
        'Pooled rows include only that model; combine both stages for the frozen all-cell summary. '
        if r.get('partial') else 'Each library-size group uses four model/suite cells. ')
    lines=['# Frozen test-A measurement','',
        scope_text+'Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.','',
        stats_table(r),'',
        markdown_table(['Cell','Contrast','Gains / losses','SR delta pp','Paired bootstrap 95% CI pp'],[
            [cell,label,f"{v['gains']}/{v['losses']}",f(v['delta'],100),interval_text(v['bootstrap_ci95'])]
            for cell,c in scopes(r) for label,v in c['comparisons'].items()]),'',
        markdown_table(['Cell','Reference','IR saved pp','Relative IR saved %'],[
            [cell,ref,f(v['ir_saved'],100),f(v['relative_ir_saved'],100)]
            for cell,c in scopes(r) for ref,v in c['ir_savings'].items()]),'',
        'Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.','']
    return '\n'.join(lines)


def _render_report_group(r):
    lines=['# 给负责人的结果说明','',
        '测试集只衡量已经冻结的任务表和统一档位，不用于重新选档。纯 C10 复用同模型同任务组的既有参照。IR 使用冻结成本系数；负节省表示开销增加。','']
    lines.append(markdown_table(['Cell']+[str(t) for t in range(10)]+['Uniform'],[
        [cell]+[r['levels'][cell][str(t)] for t in range(10)]+[r['uniform'][cell]] for cell in r['cells']]))
    lines += ['',stats_table(r),'',markdown_table(['Cell','Task violations below -5 pp','Task SR delta vs pure pp','IR saved vs uniform pp'],[
        [cell,','.join(c['task_tolerance']['task_descent']['below_minus_5pp']) or 'none',
         f(c['comparisons']['task_descent_vs_pure_c10']['delta'],100),f(c['ir_savings']['vs_uniform']['ir_saved'],100)]
        for cell,c in r['cells'].items()]),'',
        '这些结果不改变任务表。逐任务容差按冻结的 5 个百分点衡量；平均成功率不能代替每任务要求。完整配对区间、开发集与测试集差异、启动主机及数据哈希保存在 test_a.json。R15 未重复 R14 预算分配对照，因此没有分配相关性或同 IR 分配比较。','']
    return '\n'.join(lines)


def _render_handback_group(r):
    return 'Frozen measurement complete for '+', '.join(r['cells'])+'. See TEST_A.md and test_a.json; no reselection.\n'


def _render_groups(result, renderer):
    global CELLS
    complete=CELLS
    texts=[]
    try:
        for size,group in result['groups_by_library_size'].items():
            CELLS=tuple(group['cells'])
            texts.append('## Library size '+size+'\n\n'+renderer(group))
    finally:
        CELLS=complete
    return '\n\n'.join(texts)


def render_test(result):return _render_groups(result,_render_test_group)
def render_report(result):return _render_groups(result,_render_report_group)
def render_handback(result):return _render_groups(result,_render_handback_group)


def apply(stages=None, combined=False):
    require(set(os.sched_getaffinity(0)) <= set(range(2,10)) | set(range(46,54)), 'Use required CPU affinity')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide GPUs')
    reader = Reader(); reader.read(__file__); reader.read(HERE / 'analyze.py')
    p = verify_protocol(reader)
    prov = reader.json(HERE / 'provenance.json')
    for name, digest in prov['selected_output_sha256'].items():
        require(sha(reader.read(HERE / name)) == digest, f'Frozen B output changed: {name}')
    b = reader.json(HERE / 'analysis.json')
    levels = reader.json(HERE / 'levels.json'); uniforms = reader.json(HERE / 'uniform.json')
    require(set(levels) == set(uniforms) == set(CELLS), 'Incomplete frozen tables')
    scope = stage_scope([EMITTER] if stages is None else stages, levels, uniforms, reader, combined)
    for cell in CELLS:
        require(b['cells'][cell]['tolerances']['5']['levels'] == levels[cell] and
                b['cells'][cell]['tolerances']['5']['uniform'] == uniforms[cell], 'B table mismatch')
    freezes = stage_freezes(scope, prov, levels, uniforms, reader)
    caps = reader.json(HERE / 'cap_tables.json')
    templates = [reader.read(TRACE / f'r11_knob_{i}/eval500.json') for i in range(1,5)]
    require(len(set(templates)) == 1, 'R11 A templates differ')
    all_arms = {}
    for cell in scope['cells']:
        print('Read and audit A:', cell, flush=True)
        all_arms[cell] = load_a(cell, levels[cell], uniforms[cell], p, caps, reader, templates[0])
    freeze_order = check_freeze_order(all_arms, freezes)
    result = calculate_scope(all_arms, b, p, levels, uniforms, scope)
    result['episode_data'] = {c: {a: {k:getattr(ep,k).tolist() for k in ('y','n','v','m','cost')} for a,ep in arms.items()} for c,arms in all_arms.items()}
    result['provenance'] = dict(protocol_frozen_at_utc=p['frozen_at_utc'], inputs=reader.inputs,
        decision_shards=reader.shards, cpu_affinity=sorted(os.sched_getaffinity(0)), python=sys.version,
        numpy=np.__version__, bootstrap_draws=DRAWS, bootstrap_seed=SEED, pool='A', reselection=False,
        frozen_b_output_sha256=prov['selected_output_sha256'], analysis_scope=scope,
        freeze_before_test=freeze_order)
    print('Rehash inputs and verify stable shard sets', flush=True)
    reader.stable()
    names = output_names(scope)
    files = {names['test_a.json']: result}
    for name, render in (('TEST_A.md', render_test), ('REPORT.md', render_report),
                         ('ANALYSIS_TEST_A_HANDBACK.md', render_handback)):
        files[names[name]] = render(result).replace('test_a.json', names['test_a.json']).replace('TEST_A.md', names['TEST_A.md'])
    write_outputs(files, HERE)
    print(json.dumps({c: dict(sr={a:s['sr'] for a,s in x['arms'].items()},
        ir={a:s['ir'] for a,s in x['arms'].items()}, violations=x['task_tolerance']['task_descent']['below_minus_5pp'])
        for c,x in scopes(result)}, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    global EMITTER
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify-protocol','apply','combine'),nargs='?',default='verify-protocol')
    stage_args = parser.add_mutually_exclusive_group()
    stage_args.add_argument('--stage',type=Path,help='One complete model stage, or a legacy full stage')
    stage_args.add_argument('--stages',type=Path,nargs=2,help='Complete pi05 and groot stages to combine')
    args=parser.parse_args(argv)
    if args.action == 'combine' and args.stages is None:
        parser.error('combine requires --stages PI05_STAGE GROOT_STAGE')
    if args.action != 'combine' and args.stages is not None:
        parser.error('--stages requires combine')
    EMITTER=(args.stage or EMITTER).resolve()
    require(EMITTER.is_relative_to(HERE), 'Stage must remain under R15 descent')
    if args.action=='apply':apply()
    elif args.action=='combine':apply(args.stages, combined=True)
    else:
        reader=Reader();verify_protocol(reader);reader.stable()
        print('Frozen protocol verified; zero A outcome reads')

if __name__ == '__main__':
    main()
