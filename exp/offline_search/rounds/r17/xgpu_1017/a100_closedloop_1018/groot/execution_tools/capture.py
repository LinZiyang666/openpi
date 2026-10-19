"""Bounded real-input recorder; metadata-only per_step logs fail closed.

Recorder.add is also usable around policy.infer in a NEW isolated capture run.
History is an explicit QueryView history, not an inferred policy trajectory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from artifacts import digest, load_tree, save_tree, sha, tree_identity, write_json

REQUIRED = ('campaign', 'benchmark_protocol', 'task_name', 'task_id',
            'original_init_idx', 'decision_seq', 'seed', 'explicit_seed')


class Recorder:
    def __init__(self, output, *, provenance, limit=32):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.limit = limit
        self.manifest = dict(schema='xgpu.inputs.v1', provenance=provenance, rows=[])

    def add(self, observation, request, *, history=None, episode_start=None, executed_controls_since_last_query=None):
        if len(self.manifest['rows']) >= self.limit:
            return False
        if any(request.get(k) is None for k in REQUIRED):
            raise ValueError('complete request and explicit model noise seed required')
        for k in ('task_id', 'original_init_idx', 'decision_seq', 'seed', 'explicit_seed'):
            if type(request[k]) is not int or request[k] < 0:
                raise ValueError('request indices/seeds must be nonnegative integers')
        if not any(isinstance(v, np.ndarray) and v.ndim >= 3 for v in observation.values()):
            raise ValueError('raw observation images required; per_step metrics are insufficient')
        if not any('state' in k and isinstance(v, np.ndarray) for k, v in observation.items()):
            raise ValueError('raw observation state required')
        if request['decision_seq'] and history is None:
            raise ValueError('noninitial cache query requires recorded history (use {} only for explicit stateless diagnostics)')
        if request['decision_seq'] and executed_controls_since_last_query is None:
            raise ValueError('noninitial query requires an explicit execution acknowledgment count')
        if episode_start is None:
            episode_start = request['decision_seq'] == 0
        if type(episode_start) is not bool:
            raise ValueError('episode_start must be boolean')
        if executed_controls_since_last_query is None:
            executed_controls_since_last_query = 0
        if type(executed_controls_since_last_query) is not int or executed_controls_since_last_query < 0:
            raise ValueError('execution acknowledgment count must be a nonnegative integer')
        row = dict(observation=observation, request=request, history=history or {},
                   episode_start=episode_start, executed_controls_since_last_query=executed_controls_since_last_query)
        name = f'input_{len(self.manifest["rows"]):04d}'
        seal = save_tree(self.root / name, row)
        self.manifest['rows'].append(dict(file=name, sha256=seal, identity=tree_identity(row)))
        return True

    def close(self):
        if not self.manifest['rows']:
            raise ValueError('no complete recorded observations found')
        self.manifest['identity'] = digest({k:v for k,v in self.manifest.items() if k!='identity'})
        write_json(self.root / 'INPUTS.json', self.manifest)
        return self.manifest


def read_inputs(root):
    root = Path(root)
    manifest = json.loads((root / 'INPUTS.json').read_text())
    check = dict(manifest)
    identity = check.pop('identity')
    if digest(check) != identity or not manifest['rows']:
        raise ValueError('invalid input manifest')
    rows = []
    for entry in manifest['rows']:
        if Path(entry['file']).name != entry['file']:
            raise ValueError('unsafe input path')
        path = root / entry['file']
        if sha(path.with_suffix('.json')) != entry['sha256']:
            raise ValueError('input metadata changed')
        row = load_tree(path)
        if tree_identity(row) != entry['identity']:
            raise ValueError('input identity changed')
        rows.append(row)
    return manifest, rows


def from_steps(source, output, limit=32, max_lines=10000):
    source = Path(source)
    recorder = Recorder(output, provenance=dict(kind='per_step', source=str(source)), limit=limit)
    scanned = 0
    with source.open() as stream:
        for line in stream:
            scanned += 1
            if scanned > max_lines:
                break
            row = json.loads(line)
            if row.get('_kind') or row.get('accepted') is False:
                continue
            if 'observation_file' not in row:
                continue
            obs = load_tree(source.parent / row['observation_file'])
            recorder.add(obs, row['request'], history=row.get('history'),
                         episode_start=row.get('episode_start'),
                         executed_controls_since_last_query=row.get('executed_controls_since_last_query'))
            if len(recorder.manifest['rows']) >= limit:
                break
    recorder.manifest['provenance']['scanned_lines'] = scanned
    return recorder.close()


def from_archive(assets, output, benchmark, task_name, task_id, limit=3):
    """Reuse Q's SHA-bound real frames, with NEW documented diagnostic seeds."""
    assets = Path(assets)
    receipt = json.loads((assets / 'OBSERVATIONS.json').read_text())
    archive = assets / 'wire_observations.npz'
    if receipt.get('synthetic') is not False or receipt.get('recorded') is not True:
        raise ValueError('synthetic/unattested observation archive is not real evidence')
    if receipt.get('files', {}).get(archive.name) != sha(archive):
        raise ValueError('recorded archive SHA mismatch')
    recorder = Recorder(output, provenance=dict(kind='recorded_frames_new_diagnostic_seeds',
        source=str(archive), source_sha256=sha(archive), receipt=receipt,
        historical_run_reconstruction=False, history_coverage='initial queries only'), limit=limit)
    with np.load(archive, allow_pickle=False) as data:
        for i in range(min(limit, receipt['rows'])):
            obs = {k: str(data[k][i]) if k == 'prompt' else data[k][i].copy() for k in data.files}
            seed = int.from_bytes(hashlib.sha256(f'libero:7:{task_name}:0:{i}'.encode()).digest()[:4], 'little')
            # Each frame is an independent query. Frame ordinal is in the seed/provenance,
            # not falsely labelled as an actual serving trajectory.
            req = dict(campaign='xgpu_1017_recorded', benchmark_protocol=benchmark,
                task_name=task_name, task_id=task_id, original_init_idx=0,
                decision_seq=0, seed=7, explicit_seed=seed)
            recorder.add(obs, req)
    return recorder.close()


def from_raw_libero(source, output, benchmark, task_name, limit=3):
    """Consume the existing Cosmos tiny-capture format, including raw quaternion."""
    source = Path(source)
    receipt = json.loads((source / 'OBSERVATIONS.json').read_text())
    archive = source / 'raw_observations.npz'
    if receipt.get('synthetic') is not False or receipt.get('recorded') is not True:
        raise ValueError('real tiny-capture receipt required')
    if receipt.get('files', {}).get(archive.name) != sha(archive):
        raise ValueError('raw capture SHA mismatch')
    recorder = Recorder(output, provenance=dict(kind='tiny_capture', receipt=receipt,
        source_sha256=sha(archive), history_coverage='initial queries only',
        seed_policy='new diagnostic request noise seed195; original env seed retained in receipt'), limit=limit)
    with np.load(archive, allow_pickle=False) as data:
        for i in range(min(limit, receipt['rows'])):
            obs = {'observation/image': np.ascontiguousarray(data['agent_raw'][i, ::-1, ::-1]),
                   'observation/wrist_image': np.ascontiguousarray(data['wrist_raw'][i, ::-1, ::-1]),
                   'observation/state': data['state'][i].copy(),
                   'observation/cosmos_proprio': data['proprio'][i].copy(), 'prompt': str(data['prompt'][i])}
            req = dict(campaign='xgpu_1017_tiny', benchmark_protocol=benchmark,
                task_name=task_name, task_id=int(receipt['task_id']),
                original_init_idx=int(receipt['original_init_idx']), decision_seq=0,
                seed=7, explicit_seed=195)
            recorder.add(obs, req)
    return recorder.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--per-step')
    source.add_argument('--recorded-assets')
    source.add_argument('--raw-libero')
    p.add_argument('--output', required=True)
    p.add_argument('--benchmark')
    p.add_argument('--task-name')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--max-lines', type=int, default=10000)
    a = p.parse_args()
    if a.limit <= 0 or a.max_lines <= 0:
        p.error('positive bounds required')
    if a.per_step:
        result = from_steps(a.per_step, a.output, a.limit, a.max_lines)
    else:
        if not a.benchmark or not a.task_name:
            p.error('--benchmark and --task-name required for new diagnostic seeds')
        result = (from_raw_libero(a.raw_libero, a.output, a.benchmark, a.task_name, a.limit)
                  if a.raw_libero else from_archive(a.recorded_assets, a.output, a.benchmark, a.task_name, a.task_id, a.limit))
    print(json.dumps(dict(identity=result['identity'], rows=len(result['rows']))))


if __name__ == '__main__':
    main()
