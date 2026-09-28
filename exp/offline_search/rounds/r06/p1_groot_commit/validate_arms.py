"""Validate arms_p1.json + arms_rep.json: emitter, parser, CacheConfig, fit-artifact metadata, verbatim replicas.

Emits into a scratch run root (never the r06_paper root), resolving <RUN> to r06_paper so the GR00T B rows point at
the prefit artifacts. Every referenced artifact is unpickled and checked exactly like the plugin's
_load_and_fit (spec / kwargs / cell); size and sha256 recorded. Replicate rows must equal their source rows except
"name".
"""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.harness import store
import openpi.cache.config as cc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_arms import source_rows  # noqa: E402

RUN = '/home/weiland/trace_runs/os_closed_loop/r06_paper'
STORE = '/home/weiland/trace_runs/offline_search_store'
SCRATCH = Path('/tmp/p1_arm_validation')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    SCRATCH.mkdir(exist_ok=False)
    p1 = json.loads((HERE / 'arms_p1.json').read_text())
    rep = json.loads((HERE / 'arms_rep.json').read_text())
    provenance = {r['name']: r for r in json.loads((HERE / 'results' / 'arms_rep_provenance.json').read_text())}
    src = source_rows()
    # Replicates are verbatim copies of their source rows apart from the name.
    for row in rep:
        pv = provenance[row['name']]
        source = src[(pv['model'], pv['config'], pv['cell'])]['row']
        other = copy.deepcopy(row)
        other['name'] = source['name']
        assert other == source, row['name']
        assert row['name'] in (source['name'] + '_rep2', source['name'] + '_rep3')
    resolved = json.loads(json.dumps(p1 + rep).replace('<RUN>', RUN))
    spec = SCRATCH / 'spec.json'
    spec.write_text(json.dumps(resolved, indent=1))
    emit(['--run-root', str(SCRATCH), '--spec', str(spec)])
    emitted = {r['arm']: r for r in json.loads((SCRATCH / 'arms.json').read_text())}
    assert len(emitted) == len(resolved) == 36
    fits, seen = [], {}
    for row in resolved:
        e = emitted[row['name']]
        cc.load_cache_config(e['yaml'])
        opts, rest = plugin.parse_cli(['--os-method', e['method'], '--os-kwargs', json.dumps(e['kwargs']),
                                       '--os-cell', e['cell'], '--os-log-dir', str(SCRATCH / 'logs'),
                                       *e['plugin_args']])
        assert not rest and opts.os_blind and opts.os_no_shadow_native
        b = e['method'].endswith(('GrootCommitJudge', 'CommitJudge'))
        if b:
            assert e['full_model'] and e['judge'] == 'guard_only' and opts.os_policy_tail
            assert e['client_overrides']['replan_steps'] == 5 and e['replan_steps'] == 5
            assert e['kwargs']['policy_tail_gate'] == 'lifecycle' and e['kwargs']['monitor'] == 'off'
            assert e['kwargs']['base_kwargs']['serving'] == 'anchor_tail' and e['kwargs']['base_kwargs']['budget'] == 1
        else:
            assert opts.judge is None and not opts.os_policy_tail and 'full_model' not in e
            assert e['kwargs']['serving'] == 'anchor_tail' and e['kwargs']['budget'] == 1
        if e['model'] == 'groot' and b:
            assert opts.os_policy_tail_blocks == 1 and e['client_overrides']['resize_size'] == 256
            assert e['cost_ledger'] is True
        path = Path(opts.os_fit_artifact)
        if str(path) not in seen:
            with path.open('rb') as f:
                blob = pickle.load(f)
            m = blob['method']
            lib = m.base.cand_name if hasattr(m, 'base') else m.cand_name
            seen[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path),
                                   spec=blob['spec'], kwargs=blob['kwargs'], cell=blob['cell'],
                                   method_class=type(m).__name__, method_name=m.name, library=lib,
                                   library_rows=store.LibraryView(STORE, store.lib_key(blob['cell']), lib).L,
                                   bytes_per_entry=float(m.bytes_per_entry()))
            if type(m).__name__ == 'GrootCommitJudge':
                seen[str(path)].update(closed_sign=m.closed_sign, m_thr=m.m_thr, c_thr=m.c_thr)
        s = seen[str(path)]
        # The plugin's own artifact check (_load_and_fit): spec, kwargs and cell must match exactly.
        assert dict(spec=s['spec'], kwargs=s['kwargs'], cell=s['cell']) == dict(
            spec=opts.os_method, kwargs=opts.kwargs, cell=e['cell']), row['name']
        fits.append(dict(arm=row['name'], fit=str(path)))
    result = dict(PASS=True, arms_p1=len(p1), arms_rep=len(rep), emitted=len(emitted),
                  replicate_rows_verbatim_except_name=True, artifacts=list(seen.values()), arm_fit=fits)
    (HERE / 'results' / 'arms_validation.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arm_fit'}, indent=1))


if __name__ == '__main__':
    main()
