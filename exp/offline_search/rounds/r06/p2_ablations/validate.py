"""Verbatim source fields, emit_arms/config/parser, artifact metadata/size/SHA256."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.harness import store
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, STORE, artifact
import openpi.cache.config as cc


def main():
    rows = sum([json.loads((HERE / f'arms_{s}.json').read_text()) for s in ('metric', 'trigger_loo')], [])
    provenance = {r['arm']: r for r in json.loads((HERE / 'results/provenance.json').read_text())}
    assert len(rows) == len(set(r['name'] for r in rows)) == 24
    scratch = RUN / 'arm_validation'
    scratch.mkdir(parents=True, exist_ok=False)
    for row in rows:
        src = provenance[row['name']]['source_row']
        restored = copy.deepcopy(row)
        restored['name'], restored['method'] = src['name'], src['method']
        restored['kwargs'].pop('disabled_guard', None)
        args = restored['plugin_args']
        args[args.index('--os-fit-artifact') + 1] = artifact(src)
        assert restored == src, row['name']
    resolved = json.loads(json.dumps(rows).replace('<RUN>', str(RUN)))
    spec = scratch / 'spec.json'
    spec.write_text(json.dumps(resolved, indent=1) + '\n')
    emit(['--run-root', str(scratch), '--spec', str(spec)])
    emitted = json.loads((scratch / 'arms.json').read_text())
    hashes = []
    for row in emitted:
        cc.load_cache_config(row['yaml'])
        opts, rest = plugin.parse_cli(['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']),
                                       '--os-cell', row['cell'], '--os-log-dir', str(scratch / 'logs'),
                                       *row['plugin_args']])
        assert not rest and opts.os_blind and opts.os_no_shadow_native
        is_b = 'disabled_guard' in row['kwargs']
        if is_b:
            assert row['full_model'] and opts.os_policy_tail and opts.judge.mode == 'guard_only'
            assert row['client_overrides']['replan_steps'] == 5
            if row['model'] == 'groot':
                assert opts.os_policy_tail_blocks == 1 and row['client_overrides']['resize_size'] == 256
        else:
            assert opts.judge is None and not opts.os_policy_tail
        path = Path(opts.os_fit_artifact)
        with path.open('rb') as f:
            blob = pickle.load(f)
        assert {k: blob[k] for k in ('spec', 'kwargs', 'cell')} == dict(spec=opts.os_method, kwargs=opts.kwargs,
                                                                     cell=row['cell'])
        m = blob['method']
        base = m.base if is_b else m
        assert base.k == 16 and base.budget == 1 and base.serving == 'anchor_tail'
        if not is_b:
            assert hasattr(next(iter(m.tasks.values())), 'euclidean')
        lib = store.LibraryView(STORE, store.lib_key(row['cell']), base.cand_name)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 24), b''):
                h.update(block)
        hashes.append(dict(arm=row['arm'], path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest(),
                           fit_seconds=blob['fit_s'], library=base.cand_name, library_rows=lib.L,
                           bytes_per_entry=m.bytes_per_entry(), method=blob['spec']))
    out = dict(PASS=True, arms_metric=8, arms_trigger_loo=16, source_fields_exact=True,
               parser_config_and_artifact_metadata_pass=24, artifacts=hashes)
    (HERE / 'results/validation.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'results/fit_sha256.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in hashes))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
