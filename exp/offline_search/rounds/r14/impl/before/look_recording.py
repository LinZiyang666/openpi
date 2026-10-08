"""Opt-in, SHA-bound B-pool look shards. No test-A recording is eligible.

One compressed NPZ per successful real look, a manifest per server process,
and no pickle/object arrays. The manifest is atomically replaced after each
shard; interrupted unindexed shards are deliberately ignored by the reader.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from . import devset

SCHEMA = 'offline_search.dev_looks.v1'
FLAGS = ('--os-record-looks', '--os-record-look-tokens', '--os-record-look-images', '--os-record-look-shadow',
         '--os-record-look-raw-distance', '--os-record-look-cache-proposal')
CONTRACT_FLAGS = ('--os-look-contract', '--os-look-contract-sha256')


def requested(args):
    return any(v.split('=', 1)[0] in FLAGS + CONTRACT_FLAGS for v in args)


def validate_arm(row, contract):
    if not requested(row.get('plugin_args', [])):
        return False
    if (row.get('dev') is not True or row.get('init_pool') != 'B' or contract is None
            or contract.get('dev') is not True or contract.get('init_pool') != 'B'):
        raise ValueError('LOOK_RECORDING_REFUSED: only explicitly attested dev roots with init_pool=B')
    args = row.get('plugin_args', [])
    if '--os-record-looks' not in args or row.get('mode') != 'plugin':
        raise ValueError('LOOK_RECORDING_REFUSED: options require --os-record-looks on a plugin arm')
    return True


def validate_root(root):
    rows = json.loads((Path(root) / 'arms.json').read_text())
    if not rows or any(r.get('dev') is not True or r.get('init_pool') != 'B' for r in rows):
        raise ValueError('LOOK_RECORDING_REFUSED: every arm in the root must explicitly declare dev=true, init_pool=B')
    if devset.root_pool(root) != 'B':
        raise ValueError('LOOK_RECORDING_REFUSED: root is not B')


def token(contract):
    """Worker-issued episode stamp; ordinary/test-A tasks never acquire it."""
    return 'oscl-dev-look:' + hashlib.sha256(json.dumps(
        {k: contract[k] for k in ('arm', 'model', 'suite', 'manifest_sha256', 'fit_sha256')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stamp_tasks(run_gtp, pool, contract):
    if pool != 'B' or contract is None or contract.get('dev') is not True:
        raise ValueError('LOOK_RECORDING_REFUSED: recording worker must have a verified B contract')
    original = run_gtp.SweepStrategy._episodes

    def episodes(self, yaml_id, server):
        out = original(self, yaml_id, server)
        if yaml_id != contract['arm']:
            raise ValueError('LOOK_RECORDING_REFUSED: worker arm differs from recording contract')
        return [dataclasses.replace(t, extra={**t.extra, 'run_id': token(contract)}) for t in out]

    run_gtp.SweepStrategy._episodes = episodes


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def as_numpy(value, dtype=np.float32):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def retrieval_snapshot(method, topk, scores, extras):
    """Capture the full result and actual AWM anchor BEFORE MISS invalidation.

    R10Recipe returns parent-library rows, but its inner anchor addresses the
    subset. Map anchor rows once through row_subset. No metric is recomputed.
    AWM/R10 scores are negative distances; other methods retain scores only.
    """
    nodes, obj, seen = [], method, set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj)); nodes.append(obj)
        fields = vars(obj)
        obj = fields.get('inner', fields.get('base', fields.get('deployed')))
    rows, weights = np.empty(0, np.int64), np.empty(0, np.float32)
    for obj in reversed(nodes):
        anchor = vars(obj).get('_anchor')
        if isinstance(anchor, dict) and 'rows' in anchor and 'weights' in anchor:
            rows = np.array(anchor['rows'], np.int64, copy=True)
            weights = np.array(anchor['weights'], np.float32, copy=True)
            # R10 owns this mapping; wrappers can forward it via __getattr__.
            mapping = getattr(method, 'row_subset', None)
            if mapping is not None:
                rows = np.asarray(mapping)[rows].astype(np.int64)
            break
    topk, scores = np.array(topk, np.int64, copy=True), np.array(scores, np.float64, copy=True)
    is_distance = bool(nodes and any(callable(getattr(o, '_dist', None)) for o in nodes))
    distances = -scores if is_distance else np.full_like(scores, np.nan)
    top_weights = np.full(len(topk), np.nan, np.float32)
    for row, weight in zip(rows, weights, strict=True):
        top_weights[topk == row] = weight
    state = {}
    for depth, obj in enumerate(nodes):
        values = {k: v for k, v in vars(obj).items()
                  if k in ('_noprog_span', '_last_progress', '_last_top1', '_burst_left',
                           '_vision_progress', '_guard_identity', 'noprog_n', 'prog_eps',
                           'stuck_thr', 'burst', 'disabled_guards', 'disabled_mask',
                           '_guard_steps', '_trigger_steps')}
        state[str(depth)] = {k: sorted(v) if isinstance(v, set) else json_value(v) for k, v in values.items()}
        memo = vars(obj).get('_s')
        if isinstance(memo, dict):
            state[str(depth)]['memo'] = json_value({k: memo[k] for k in
                ('stuck_n', 'prog', 'flag', 'burst_end', 'ret_end') if k in memo})
    return dict(topk=topk, scores=scores, distances=distances, weights=top_weights,
                donor_rows=rows, donor_weights=weights,
                guard_state=dict(method=state, extras=json_value(extras or {})),
                distance_semantics='negative Result.scores (AWM/R10)' if is_distance else 'unavailable; inspect scores')


class LookWriter:
    def __init__(self, opts, model):
        path = Path(opts.os_look_contract)
        if not path.is_file() or devset.sha(path) != opts.os_look_contract_sha256:
            raise ValueError('LOOK_RECORDING_REFUSED: missing or changed server contract')
        self.contract = json.loads(path.read_text())
        c = self.contract
        if (c.get('schema') != SCHEMA or c.get('dev') is not True or c.get('init_pool') != 'B'
                or c.get('model') != model or c.get('cell') != opts.os_cell or c.get('method') != opts.os_method):
            raise ValueError('LOOK_RECORDING_REFUSED: server contract is not this B-pool arm')
        self.pairs = {tuple(p) for p in c['pairs']}
        if not self.pairs or any(i in c['subset']['excluded_inits_by_task'][str(t)] for t, i in self.pairs):
            raise ValueError('LOOK_RECORDING_REFUSED: empty/overlapping B selection')
        self.opts = opts
        self.shadow = opts.os_record_look_shadow
        arm_dir = Path(c['recording_arm_dir']).resolve()
        if not Path(opts.os_log_dir).resolve().is_relative_to(arm_dir):
            raise ValueError('LOOK_RECORDING_REFUSED: server log directory is outside the attested dev arm')
        self.directory = Path(opts.os_log_dir) / 'looks'
        # Never escape the coordinator's per-arm server directory through a symlink.
        if self.directory.resolve().parent != Path(opts.os_log_dir).resolve():
            raise ValueError('LOOK_RECORDING_REFUSED: looks directory escaped server log directory')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.id = uuid.uuid4().hex
        self.manifest = self.directory / f'manifest_{self.id}.json'
        self.lock = threading.Lock()
        self.data = dict(schema=SCHEMA, dev=True, init_pool='B', contract=c,
                         contract_sha256=opts.os_look_contract_sha256, shards=[],
                         fields=dict(key_v0='vision_0 key_builder output: float32, 32768; agentview/third-person',
                                     key_v1='vision_1 key_builder output: float32, 32768; wrist',
                                     rs='normalized live robot_state key', raw_state='wire state, float32',
                                     action='executed normalized model chunk, H x 32',
                                     wire_actions='served client action chunk after output transform',
                                     images='nearest-neighbor downsample of wire RGB, uint8; no extra encoder',
                                     tokens='key-builder stage-1 slice, float16; independent of --os-tokens',
                                     weights='actual AWM donor weights, NaN when unavailable'),
                         options=dict(tokens=opts.os_record_look_tokens, image_size=opts.os_record_look_images,
                                      shadow=self.shadow))
        if getattr(opts, "os_record_look_raw_distance", False):
            self.data["options"]["raw_distance"] = True
            self.data["fields"]["raw_distance"] = "minimum geometric inner.base._dist(q)[7]; float64"
        if getattr(opts, "os_record_look_cache_proposal", False):
            self.data["options"]["cache_proposal"] = True
            self.data["fields"]["cache_proposal"] = "corrected normalized cache proposal before policy replacement; first ten controls"
        self._flush()

    def _flush(self):
        temp = self.manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(self.data, sort_keys=True, indent=1) + '\n')
        temp.replace(self.manifest)

    def validate_episode(self, extra, task, init):
        if (extra.get('run_id') != token(self.contract) or (task, init) not in self.pairs
                or not str(extra.get('task_uid', '')).startswith(self.contract['arm'] + ':eval:')):
            raise ValueError('LOOK_RECORDING_REFUSED: episode lacks matching B worker stamp/selection/arm')

    def capture(self, session, ctx, topk, scores, extras):
        out = retrieval_snapshot(session.method, topk, scores, extras)
        if getattr(self.opts, "os_record_look_raw_distance", False):
            if "os_r13_raw_distance" in (extras or {}):
                distance = extras["os_r13_raw_distance"]
            else:
                from .plugin import OnlineQueryView
                view = OnlineQueryView(session, session.step, session.ep.task_id, session.ep)
                base = session.method.inner.base
                distance = np.min(np.asarray(base._dist(view)[7], np.float64))
            out["raw_distance"] = np.array(distance, np.float64)
        if getattr(self.opts, "os_record_look_cache_proposal", False):
            out["cache_proposal"] = np.array(session._dec["served"][:10], np.float32, copy=True)
        for name, field in (('key_v0', 'vision_0'), ('key_v1', 'vision_1'), ('rs', 'robot_state')):
            out[name] = as_numpy(ctx.query_keys[field]).reshape(-1)
        if any(out[k].shape != (32768,) for k in ('key_v0', 'key_v1')):
            raise ValueError('recording expects the deployed 32768-d per-view keys')
        if self.opts.os_record_look_tokens:
            raw = session.kb._slice()
            for name, field in (('tok_v0', 'vision_0'), ('tok_v1', 'vision_1')):
                out[name] = as_numpy(raw[field], np.float16).reshape(-1, 2048)
        size = self.opts.os_record_look_images
        if size:
            for name, field in (('img0', 'observation/image'), ('img1', 'observation/wrist_image')):
                im = np.asarray(session.cur_obs[field], np.uint8)
                y = np.minimum((np.arange(size) * im.shape[0] / size).astype(int), im.shape[0] - 1)
                x = np.minimum((np.arange(size) * im.shape[1] / size).astype(int), im.shape[1] - 1)
                out[name] = im[y[:, None], x[None, :]].copy()
        return out

    def write(self, session, decision, wire, adapter):
        if not decision.get('vision', True):
            return
        snap = decision['look_snapshot']
        step = decision['step']
        action = session.b_aex.a[step].copy()
        arrays = {k: v for k, v in snap.items() if isinstance(v, np.ndarray)}
        guard = {**snap['guard_state'], 'hit': bool(decision.get('hit', True)),
                 'judge': decision.get('judge'), 'tau': json_value(decision.get('tau')),
                 'run': decision.get('run'), 'burst_left': getattr(session, 'burst_left', 0),
                 'blind_age_before': session._age_before, 'look_reason': session._look_reason}
        meta = dict(schema=SCHEMA, dev=True, init_pool='B', arm=self.contract['arm'],
                    model=self.contract['model'], suite=self.contract['suite'],
                    **session.ep_meta, episode_id=session.ident.get('episode_id', -1),
                    step=step, ctx_step=decision['ctx_step'], library=decision['lib'],
                    camera_mode=getattr(session, '_camera_mode', 'full'),
                    distance_semantics=snap['distance_semantics'],
                    shadow_source='served_policy' if not decision.get('hit', True) else 'private_noise_same_stage1')
        arrays.update(step=np.array(step, np.int64), task_id=np.array(meta['task_id'], np.int64),
                      init=np.array(meta['init'], np.int64), raw_state=session.b_raw.a[step].copy(),
                      action=action, wire_actions=np.array(wire['actions'], copy=True),
                      hit=np.array(decision.get('hit', True)),
                      meta=np.array(json.dumps(json_value(meta), sort_keys=True)),
                      guard_state=np.array(json.dumps(json_value(guard), sort_keys=True)))
        interval = getattr(self.opts, 'os_groot_path_check_every', 0)
        if interval and self.contract['model'] == 'groot' and sum(session.has_vision) % interval == 1 % interval:
            from .lookhalf_numerics import record
            numerical, info = record(session, adapter)
            arrays.update(numerical)
            arrays['groot_numerical_path'] = np.array(json.dumps(info, sort_keys=True))
        if getattr(session.rt, 'lookhalf', None) is not None:
            arrays['camera_availability'] = np.array(session._camera_work.get('availability', [True, True]), bool)
            arrays['lookhalf'] = np.array(json.dumps(json_value(decision.get('lookhalf', {})), sort_keys=True))
            from .lookhalf_cameras import ledger
            costs = ledger(self.contract['model'],vision=True,
                wrist_first=getattr(session,'_camera_mode','full') == 'wrist_only',
                completions=session._camera_work.get('completions',0),miss=not bool(decision.get('hit',True)))
            arrays['lookhalf_cost'] = np.array(json.dumps(costs,sort_keys=True))
        if getattr(session.rt, 'cache_ablate', None) is not None:
            from .cache_ablate_cameras import ledger
            costs = ledger(self.contract['model'], session.rt.cache_ablate, vision=True,
                           work=session._camera_work, hit=bool(decision.get('hit', True)))
            arrays['camera_availability'] = np.array(session._camera_work['availability'], bool)
            arrays['camera_encoded_count'] = np.array(costs['camera_encoded_count'])
            arrays['cache_ablate_cost'] = np.array(json.dumps(costs, sort_keys=True))
        if self.shadow:
            arrays['shadow_policy'] = (action.copy() if not decision.get('hit', True)
                                       else shadow_chunk(session, adapter))
        with self.lock:
            name = f'look_{self.id}_{len(self.data["shards"]):07d}.npz'
            path = self.directory / name
            temp = path.with_suffix('.tmp')
            with temp.open('xb') as f:
                np.savez_compressed(f, **arrays)
            temp.replace(path)
            self.data['shards'].append(dict(file=name, sha256=devset.sha(path), uid=meta['uid'],
                                           attempt=meta['attempt'], step=step, task_id=meta['task_id'], init=meta['init']))
            self._flush()
        session._look_stage1 = None


def shadow_chunk(session, adapter):
    """Same stage-1 observation, explicit private noise, no global RNG draws."""
    if adapter is None:
        raise RuntimeError('recording shadow requires a stage adapter')
    if adapter.fake:
        return np.array(adapter.policy._osp_shadow_chunk(session.cur_obs), np.float32, copy=True)
    import torch
    from .plugin import ep_seed
    p, stage1 = adapter.policy, session._look_stage1
    with adapter.lock, torch.inference_mode():
        if session.rt.model == 'pi05':
            device = stage1.state.device
            gen = torch.Generator(device=device).manual_seed(ep_seed(session.ep.seed, f'shadow:{session.step}'))
            noise = torch.randn((1, session.rt.H, 32), device=device, dtype=torch.float32, generator=gen)
            stage2 = p._stage2_fn(stage1)
            if p._stage_config is not None and p._stage_config.needs_relocation:
                stage2 = stage2.to(p._stage3_device)
            out = p._stage3_fn(stage2, noise=noise, num_steps=session.miss_steps)
            return as_numpy(out.action_chunk).reshape(session.rt.H, 32)
        runner = p._runner
        with runner.session():
            stage2 = runner.run_stage2_llm(stage1)
            gen = torch.Generator(device=stage2.backbone_features.device).manual_seed(
                ep_seed(session.ep.seed, f'shadow:{session.step}'))
            noise = runner.sample_noise(stage2, generator=gen)
            return as_numpy(runner.run_stage3(stage2, noise=noise).action_pred).reshape(session.rt.H, 32)


def read_looks(directory, *, verify=True):
    """Yield independent dictionaries; always use allow_pickle=False and verify B."""
    directory = Path(directory)
    manifests = [directory] if directory.is_file() else sorted(directory.glob('manifest_*.json'))
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        if data.get('schema') != SCHEMA or data.get('dev') is not True or data.get('init_pool') != 'B':
            raise ValueError('LOOK_RECORDING_REFUSED: reader accepts only v1 dev B recordings')
        for shard in data['shards']:
            name = shard['file']
            if Path(name).name != name:
                raise ValueError('unsafe look shard path')
            path = manifest.parent / name
            if path.resolve().parent != manifest.parent.resolve():
                raise ValueError('look shard escaped recording directory')
            if verify and devset.sha(path) != shard['sha256']:
                raise ValueError('look shard SHA mismatch')
            with np.load(path, allow_pickle=False) as z:
                out = {k: z[k].copy() for k in z.files}
            out['meta'] = json.loads(str(out['meta']))
            out['guard_state'] = json.loads(str(out['guard_state']))
            if out['meta'].get('init_pool') != 'B' or out['meta'].get('dev') is not True:
                raise ValueError('foreign pool in look shard')
            yield out


def prepare(source, destination, names=None, *, tokens=False, images=0, shadow=False):
    """Fresh dev root for the coordinator; source fits/manifests remain frozen."""
    from .ops.h100.assets import flag
    from .ops.remote.run_gtp_subset import load_manifest
    source, destination = Path(source).resolve(), Path(destination).resolve()
    validate_root(source)
    if destination.exists():
        raise FileExistsError('fresh destination required')
    rows = json.loads((source / 'arms.json').read_text())
    names = names or [r['arm'] for r in rows]
    if set(names) - {r['arm'] for r in rows} or len(set(names)) != len(names):
        raise ValueError('unknown/duplicate arm')
    if not 0 <= images <= 256:
        raise ValueError('image size must be 0..256')
    prepared, files = [], {}
    for row in rows:
        if row['arm'] not in names:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_]+', row['arm']):
            raise ValueError('unsafe arm name')
        row = json.loads(json.dumps(row).replace('<RUN>', str(source)))
        contract = devset.arm_contract(source, row, load_manifest(row['manifest']))
        if contract is None:
            raise ValueError('LOOK_RECORDING_REFUSED: missing dev contract')
        if requested(row.get('plugin_args', [])):
            raise ValueError('source already has look options; use its root directly')
        for field in ('yaml', 'matrix'):
            old = Path(row[field])
            new = destination / 'config' / old.name
            content = old.read_text().replace('<RUN>', str(source))
            if field == 'matrix':
                content = content.replace(str(row['yaml']), str(destination / 'config' / Path(row['yaml']).name))
            files[new] = content
            if field == 'matrix':
                row['matrix'] = str(new)
        row['yaml'] = str(destination / 'config' / Path(row['yaml']).name)
        row['plugin_args'] += ['--os-record-looks']
        if tokens: row['plugin_args'] += ['--os-record-look-tokens']
        if images: row['plugin_args'] += ['--os-record-look-images', str(images)]
        if shadow:
            row['plugin_args'] += ['--os-record-look-shadow']
            row['full_model'] = True
        assert flag(row['plugin_args'], '--os-fit-artifact')
        prepared.append(row)
    destination.mkdir(parents=True)
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    (destination / 'arms.json').write_text(json.dumps(prepared, indent=1) + '\n')
    return prepared


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('source'); p.add_argument('destination'); p.add_argument('arms', nargs='*')
    p.add_argument('--tokens', action='store_true'); p.add_argument('--images', type=int, default=0)
    p.add_argument('--shadow', action='store_true')
    r = sub.add_parser('inspect'); r.add_argument('directory')
    a = ap.parse_args()
    if a.action == 'prepare':
        rows = prepare(a.source, a.destination, a.arms, tokens=a.tokens, images=a.images, shadow=a.shadow)
        print(f'RECORDING_DEV_PREPARED root={a.destination} arms={len(rows)} init_pool=B')
    else:
        records = list(read_looks(a.directory))
        print(json.dumps(dict(looks=len(records), episodes=len({(r['meta']['uid'], r['meta']['attempt']) for r in records}),
                              init_pool='B', schema=SCHEMA)))


if __name__ == '__main__':
    main()
