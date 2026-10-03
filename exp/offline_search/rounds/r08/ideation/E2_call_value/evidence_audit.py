"""Bounded read-only R8 E2 evidence audit; no inference or simulator imports.

Reads the 16 CU/CT non-test profiles, one trace file per inference cell, and
one P3 episode per model. Writes only this directory and assigned scratch.
"""
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from exp.offline_search.rounds.r07.c4_profile.common import accepted_arm


HERE = Path(__file__).resolve().parent
SCRATCH = Path('/tmp/r8_E2_call_value')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
start = time.monotonic()
sources = []


def source(path):
    path = Path(path)
    st = path.stat()
    sources.append(dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    return path


support = []
for model in ('pi05', 'groot'):
    for suite in ('l10', 'spatial'):
        for size, rho in ((50, 30), (500, 18)):
            for variant in ('CU', 'CT'):
                arm = f'r7_{model}_{suite}_{size}_{variant}{rho}_profile'
                root = RUNS / 'r07_profile_bval1/runs' / arm
                for path in root.glob('server_*/decisions_*.jsonl'):
                    source(path)
                source(root / 'client/journal.jsonl')
                eps, audit = accepted_arm(root)
                decisions = [d for e in eps for d in e['decisions']]
                fresh = [d for d in decisions if d.get('extras', {}).get('os_c_fresh') == 1]
                assert len(fresh) == sum(bool(d['vision']) for d in decisions)
                counts = Counter()
                for d in fresh:
                    x = d['extras']
                    p, z = x['os_c_p'], int(not d['hit'])
                    assert z == int(x['os_c_coin'] < p)
                    counts['anchors'] += 1
                    counts['supported'] += int(0 < p < 1)
                    counts['p_zero'] += int(p == 0)
                    counts['p_one'] += int(p == 1)
                    counts['supported_calls'] += int(0 < p < 1 and z == 1)
                    counts['supported_cache'] += int(0 < p < 1 and z == 0)
                    counts['forced_stall_calls'] += int(bool(x['os_c_stall_call']))
                    if x.get('os_c3_dev_entry'):
                        counts['entries'] += 1
                        counts['entry_supported'] += int(0 < p < 1)
                        counts['entry_p_zero'] += int(p == 0)
                        counts['entry_p_one'] += int(p == 1)
                        counts['entry_nominal_one'] += int(x['os_c_nominal_p'] == 1)
                support.append(dict(arm=arm, cell=f'{model}_{suite}_{size}', variant=variant,
                                    audit=audit, decisions=len(decisions), **counts))

# Metadata only: do not read multi-MB embeddings or image tensors.
trace_headers = []
trace_root = Path('/home/weiland/trace_runs/dual_20260923/runs')
for model in ('pi05', 'groot'):
    for suite in ('l10', 'sp'):
        root = trace_root / f'tr_{model}_{suite}_inf/trace'
        path = next(root.rglob('*.h5'), None)
        if path is None:
            path = next(root.rglob('*.hdf5'))
        fields = {}
        with h5py.File(source(path), 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    fields[name] = dict(shape=list(obj.shape), dtype=str(obj.dtype))
            f.visititems(visit)
            attrs = {key: str(value)[:250] for key, value in f.attrs.items()}
        gt = [name for name in fields if any(token in name.lower() for token in
              ('sim_state', 'qpos', 'qvel', 'contact', 'body_xpos', 'object_pose'))]
        trace_headers.append(dict(path=str(path), fields=fields, attributes=attrs,
                                  simulator_field_name_matches=gt))

p3 = []
for model in ('pi05', 'groot'):
    root = RUNS / 'r06_p3_pilot/runs' / f'r6p3v2_{model}_l10_50_A_r0'
    control_path = sorted((root / 'client_telemetry').glob('*/controls.jsonl'))[0]
    snapshots = sorted(control_path.parent.glob('step_*.npz'))
    snapshot = snapshots[0]
    with np.load(source(snapshot), allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata_json']))
        skipped = json.loads(str(z['controller_skipped_json']))
        snapshot_fields = {key: dict(shape=list(z[key].shape), dtype=str(z[key].dtype)) for key in z.files}
    physical = None
    decision_rows = []
    controls = 0
    with source(control_path).open() as f:
        for line in f:
            row = json.loads(line)
            if row['ev'] == 'control':
                controls += 1
                if physical is None:
                    after = row['after']
                    physical = dict(fields=list(after), event_labels=after.get('event_labels'),
                                    predicates=after.get('predicates'),
                                    actuator_trace_missing=row.get('actuator_trace_missing'))
            if row['ev'] == 'decision':
                decision_rows.append(dict(step=row['decision_step'], control=row['control'],
                                          marker=row['marker'], snapshot=row.get('snapshot')))
    # The hashed episode directory name is shared by server NPZ and client.
    folder = next(root.glob('server_*/p3_inputs/' + control_path.parent.name))
    divergence = []
    payload_fields = {}
    for path in sorted(folder.glob('step_*.npz')):
        with np.load(source(path), allow_pickle=False) as z:
            if not payload_fields:
                payload_fields = {key: dict(shape=list(z[key].shape), dtype=str(z[key].dtype)) for key in z.files}
            if 'diagnostic_cache_chunk' not in z:
                continue
            tail = z['executed_chunk'][:5, :7].astype(float)
            fresh = z['diagnostic_cache_chunk'][:5, :7].astype(float)
            policy = z['policy_chunk'][:5, :7].astype(float)
            rms = lambda a, dims: float(np.sqrt(np.mean((a[:, dims] - policy[:, dims]) ** 2)))
            divergence.append(dict(step=int(path.stem.split('_')[-1]),
                                   tail_rms=rms(tail, slice(7)), fresh_rms=rms(fresh, slice(7)),
                                   tail_motion_rms=rms(tail, slice(6)), fresh_motion_rms=rms(fresh, slice(6))))
    p3.append(dict(model=model, episode_directory=str(control_path.parent),
                   snapshot_bytes=snapshot.stat().st_size, snapshot_count=len(snapshots),
                   snapshot_fields=snapshot_fields, snapshot_metadata=metadata,
                   skipped_controller=skipped, physical=physical, controls=controls,
                   decisions=decision_rows, input_fields=payload_fields,
                   illustrative_single_episode_divergence=divergence))

result = dict(scope='16 non-test profile arms; 4 old trace headers; 2 P3 episodes only',
              support=support, trace_headers=trace_headers, p3=p3,
              elapsed_s=time.monotonic()-start, sources=sources)
(HERE / 'evidence_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(elapsed_s=result['elapsed_s'],
                     support=support,
                     trace_header_fields=[dict(path=r['path'], fields=len(r['fields']),
                                               gt_matches=r['simulator_field_name_matches']) for r in trace_headers],
                     p3=[dict(model=r['model'], snapshot_bytes=r['snapshot_bytes'],
                              snapshots=r['snapshot_count'], controls=r['controls'],
                              skipped=r['skipped_controller'],
                              event_labels=r['physical']['event_labels'],
                              restore_certified=r['snapshot_metadata']['restore_certified'],
                              blind_diagnostics=len(r['illustrative_single_episode_divergence'])) for r in p3]), indent=2))
