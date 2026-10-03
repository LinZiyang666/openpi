"""Bounded, read-only inspection for R8 E5; no serving/simulator imports.

Reads four HDF5 first-step metadata trees, twenty NPZ central directories,
and through the first three active controls of four P3 episodes. Writes only E5 evidence.
"""
import hashlib
import json
from pathlib import Path
import time
import zipfile
import zlib

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs')


def first(paths):
    return next(iter(sorted(paths)), None)


def zip_info(path):
    if path is None:
        return None
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        return dict(path=str(path), bytes=path.stat().st_size,
                    members=[dict(name=x.filename, raw=x.file_size, stored=x.compress_size)
                             for x in members])


def accepted_episode(arm):
    with (arm / 'client/journal.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('accepted') and row.get('status') in ('done', 'failed'):
                return row
    raise ValueError('No accepted episode: ' + str(arm))


def inspect_p3(cell):
    arm = ROOT / 'os_closed_loop/r06_p3_pilot/runs' / ('r6p3v2_' + cell.replace('_spatial', '_sp') + '_50_A_r0')
    accepted = accepted_episode(arm)
    uid = accepted.get('task_uid')
    attempt = accepted.get('attempt')
    key = hashlib.sha256(uid.encode()).hexdigest()[:24]
    directory = arm / 'client_telemetry' / f'{key}_a{attempt}'
    path = directory / 'controls.jsonl'
    controls = []
    with path.open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('ev') == 'control' and row.get('decision_step') is not None:
                after = row['after']
                controls.append(dict(bytes=len(line), zlib_bytes=len(zlib.compress(line, 1)),
                    control=row['control'], decision=row['decision_step'],
                    fields=list(after), numeric_shapes={k: list(np.asarray(v).shape)
                        for k, v in after.items() if isinstance(v, list) and k != 'contacts'},
                    predicates=after.get('predicates'), event_labels=after.get('event_labels'),
                    actuator_trace_missing=row.get('actuator_trace_missing'),
                    physics_substeps=len(row.get('actuator_ctrl_each_physics_step', [])),
                    entity_ids=after.get('entity_ids'),
                    obs_fields=list(after.get('observation_numeric', {}))))
                if len(controls) == 3:
                    break
    files = [p for p in directory.iterdir() if p.is_file()]
    snapshots = [p for p in files if p.suffix == '.npz']
    server_dir = first(arm.glob(f'server_*/p3_inputs/{key}_a{attempt}'))
    server_files = list(server_dir.glob('*.npz')) if server_dir else []
    return dict(arm=str(arm), accepted_uid=uid, accepted_attempt=attempt,
        journal_keys=list(accepted), client_bytes=sum(p.stat().st_size for p in files),
        controls_bytes=path.stat().st_size, snapshot_count=len(snapshots),
        snapshot_bytes=sum(p.stat().st_size for p in snapshots), sample_controls=controls,
        sample_snapshot=zip_info(first(snapshots)), server_decisions=len(server_files),
        server_decision_bytes=sum(p.stat().st_size for p in server_files),
        sample_server_input=zip_info(first(server_files)),
        ordinary_input=zip_info(first(arm.glob('server_*/inputs/*.npz'))))


def inspect_trace(model, suite):
    label = 'sp' if suite == 'spatial' else suite
    arm = ROOT / 'dual_20260923/runs' / f'tr_{model}_{label}_inf'
    path = first(arm.glob('trace/*/*.h5'))
    with h5py.File(path, 'r') as f:
        steps = sorted(k for k in f if k.startswith('step_'))
        datasets = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(dict(name=name, shape=list(obj.shape), dtype=str(obj.dtype),
                    raw_bytes=obj.size * obj.dtype.itemsize, stored_bytes=obj.id.get_storage_size()))
        f[steps[0]].visititems(visit)
        return dict(path=str(path), bytes=path.stat().st_size, decisions=len(steps),
                    sample_step=steps[0], datasets=datasets)


def main():
    started = time.perf_counter()
    report = dict(scope='Four first-step HDF5 metadata trees; four accepted P3 episodes, '
                       'through first three active controls; eight ordinary R7 NPZ central directories.',
                  trace={}, p3={}, r7={})
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'spatial'):
            cell = f'{model}_{suite}'
            report['trace'][cell] = inspect_trace(model, suite)
            report['p3'][cell] = inspect_p3(cell)
            for size in (50, 500):
                arm = ROOT / 'os_closed_loop/r07_profile_bval1/runs' / f'r7_{cell}_{size}_A_profile'
                report['r7'][f'{cell}_{size}'] = zip_info(first(arm.glob('server_*/inputs/*.npz')))
    report['elapsed_seconds'] = time.perf_counter() - started
    (HERE / 'evidence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(elapsed_seconds=report['elapsed_seconds'],
        trace={k: dict(bytes=v['bytes'], decisions=v['decisions'],
            image_fields=[d['name'] for d in v['datasets'] if 'image' in d['name']],
            sim_fields=[d['name'] for d in v['datasets'] if any(s in d['name'] for s in ('sim_', 'contact', 'body_x'))])
            for k, v in report['trace'].items()},
        p3={k: {name: v[name] for name in ('client_bytes', 'controls_bytes', 'snapshot_count',
                'snapshot_bytes', 'server_decisions', 'server_decision_bytes', 'sample_controls')}
            for k, v in report['p3'].items()},
        r7={k: None if v is None else dict(bytes=v['bytes'],
            image_fields=[m['name'] for m in v['members'] if any(s in m['name'] for s in ('image', 'img'))])
            for k, v in report['r7'].items()}), indent=2))


if __name__ == '__main__':
    main()
