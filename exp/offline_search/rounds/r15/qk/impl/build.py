"""Reproducible CPU export + same-library R10 layers 1--3, frozen periodic caps."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import re
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r10.recipe import fitting
from .encoder import TinySpatial, FeatureExport, FrozenEncoder, inputs, sha
from .recipe import LightKeyKnob
from .boundary import library_boundary
from .fit_math import donor_scores

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
ASTRA = HERE.parent / 'astra'
STORE = Path('/home/weiland/trace_runs/offline_search_store')
DERIVED = STORE / 'derived/r15_qk/impl'
SPEC = 'exp.offline_search.rounds.r15.qk.impl.recipe:LightKeyKnob'


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def baseline(size, level):
    if size == 200:
        rec = json.loads((REPO/'exp/offline_search/rounds/r15/descent/fits.json').read_text())[f'pi05_l10_{size}'][level]
        path = Path(rec['path'])
        assert sha(path) == rec['sha256']
    else:
        suffix = 'off' if level == 'off' else 'periodic_ir25'
        path = REPO/f'exp/offline_search/rounds/r11/knob/artifacts/r11_pi05_l10_{size}_{suffix}.pkl'
    return path, pickle.loads(path.read_bytes())


def default_config(size, out):
    source = ASTRA/f'pi05_l10_{size}/cnn_action_cpu.pt'
    training = source.with_name('cnn_action_cpu_train.json')
    record = json.loads(training.read_text())
    ck = torch.load(source, map_location='cpu', weights_only=True)
    if ck['epochs'] != 80 or ck['objective'] != 'action' or ck['seed'] != 20261005 or record['guard']['device'] != 'cpu':
        raise ValueError('partial or wrong training artifact')
    teacher = REPO/f'exp/offline_search/rounds/r10/recipe/artifacts/r10_recipe_pi05_l10_{size}.pkl'
    if record['provenance']['artifact_sha256'] != sha(teacher):
        raise ValueError('student training teacher changed')
    sd = ck['state_dict']
    model = TinySpatial(sd['pm'], sd['ps'], sd['rm'], sd['rs'])
    model.load_state_dict(sd)
    model.eval()
    export = torch.jit.freeze(torch.jit.script(FeatureExport(model).eval()))
    export_path = out/'student.ts'
    export.save(str(export_path))
    parity = []
    rng = np.random.default_rng(20261005)
    for task in range(10):
        for mode in range(3):
            images = rng.integers(0, 256, (2, 224, 224, 3), np.uint8)
            state = rng.normal(size=8).astype(np.float32) * (1 if mode < 2 else 20)
            with torch.inference_mode():
                args = inputs(images, state, task)
                a, b = model(*args)[0].numpy(), export(*args).numpy()
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
            parity.append(dict(task=task, perturbation=mode, max_abs=float(np.max(np.abs(a-b)))))
    spec = dict(factory='exp.offline_search.rounds.r15.qk.impl.encoder:CNNStudent',
        kwargs=dict(model_path=str(export_path), model_sha256=sha(export_path),output_index=None),
        device='cpu', threads=1, feature_shape=[2, 64], feature_kind='light_symmetric',
        epsilon=.0140 if size == 50 else .0144, cost_basis='astra COST.md pi05 graph basis, rounded owner coefficients',
        cell=f'pi05_l10_{size}', completed_epochs=80, library_only=True,
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        source_training=str(training), source_training_sha256=sha(training),
        teacher=str(teacher), teacher_sha256=sha(teacher), export_parity=parity,
        input_recipe='agentview,wrist uint8 224; indices 4*j ->56; /255 then *2-1; normalized state8; task id',
        candidate_name='cnn_action_cpu',
        training_provenance=record['provenance'])
    spec['representation_sha256'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


class FeatureLibrary(fitting._Library):
    def __init__(self, *args, features, **kwargs):
        super().__init__(*args, **kwargs)
        if features.shape != (self.L, 2, 64):
            raise ValueError('student features do not align with selected rows')
        self._a.update(key_v0=features[:, 0], key_v1=features[:, 1])


def build(size, config=None, tag=None):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if tag is not None and not re.fullmatch(r'[a-z0-9_]+',tag):
        raise ValueError('invalid artifact tag')
    out = HERE/'artifacts'/tag/f'pi05_l10_{size}' if tag else HERE/f'artifacts/pi05_l10_{size}'
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(config).read_text()) if config else default_config(size, out)
    if spec['cell'] != f'pi05_l10_{size}' or not spec['library_only'] or spec['completed_epochs'] != spec.get('declared_epochs',80):
        raise ValueError('wrong-cell or partial frozen student')
    subset = REPO/f'exp/offline_search/rounds/r10/subsets/pi05_l10_{size}.json'
    sub = json.loads(subset.read_text())
    train_prov = spec.get('training_provenance', {})
    if train_prov.get('subset_sha256') != sha(subset):
        raise ValueError('student trained on another episode subset')
    for field in ('source_checkpoint','source_training'):
        if field in spec and sha(spec[field]) != spec[field+'_sha256']:
            raise ValueError(f'changed frozen student provenance: {field}')
    if train_prov.get('artifact') and sha(train_prov['artifact']) != train_prov['artifact_sha256']:
        raise ValueError('changed library-only training teacher')
    if spec.get('candidate_manifest') and sha(spec['candidate_manifest'])!=spec['candidate_manifest_sha256']:
        raise ValueError('changed frozen PILOT manifest')
    encoder = FrozenEncoder(spec)
    chosen = sub['episode_ids_by_task']
    libdir = STORE/'library/pi05_l10/bpool_cs'
    dataout = DERIVED/tag/f'pi05_l10_{size}' if tag else DERIVED/f'pi05_l10_{size}'
    dataout.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with library_boundary(STORE, libdir) as reads:
        lib = fitting._Library(STORE, 'pi05_l10', 'bpool_cs', chosen)
        tokrows = np.load(libdir/'tok/rows.npy')
        if len(np.unique(tokrows)) != len(tokrows):
            raise ValueError('duplicate token row IDs')
        inv = np.full(lib.parent.L, -1, int)
        inv[tokrows] = np.arange(len(tokrows))
        if np.any(inv[lib.rows] < 0):
            raise ValueError('missing selected library RGB')
        images = [np.load(libdir/f'tok/img{cam}.npy', mmap_mode='r') for cam in range(2)]
        if any(im.shape[1:] != (224,224,3) or im.dtype != np.uint8 for im in images):
            raise ValueError('wrong library camera format')
        image_hash = [hashlib.sha256(), hashlib.sha256()]
        features = []
        history, episode_start, previous_episode = [], 0, None
        for i, row in enumerate(lib.rows):
            episode = int(lib.episode[i])
            if episode != previous_episode:
                history, episode_start, previous_episode = [], i, episode
            e = next(e for e in lib.episodes if int(lib.parent.episode[e['start']]) == episode)
            n = i - episode_start
            look = SimpleNamespace(img0=np.array(images[0][inv[row]]), img1=np.array(images[1][inv[row]]),
                rs=lib.rs[i], task_id=int(lib.task_id[i]), step=int(lib.step[i]),
                raw_state=lib.raw_state[i] if lib.has('raw_state') else None,
                hist_looks=tuple(history), hist_images=tuple((h['img0'],h['img1']) for h in history),
                hist_features=tuple(h['features'] for h in history),
                last_policy_output=lib.action[i-1] if n else None,
                last_policy_wire_output=None, instruction=e.get('task',''),
                hist_a_exec=lib.action[episode_start:i], hist_hit=np.zeros(n, np.int8),
                hist_has_look=np.ones(n, bool))
            # Single-frame default ignores history. Generic frozen students must
            # declare a causal library look-sequence adapter in their factory.
            for cam in range(2):
                image_hash[cam].update(getattr(look, f'img{cam}').tobytes())
            feature = encoder.encode_raw(look)
            features.append(feature)
            history.append(dict(step=look.step,img0=look.img0,img1=look.img1,features=feature))
        features = np.stack(features)
        if features.shape[1:] != (2,64) and np.prod(features.shape[1:]) == 128:
            spec.update(raw_feature_shape=list(spec['feature_shape']),feature_shape=[2,64])
            features = features.reshape(len(features),2,64)
        elif features.shape[1:] != (2,64):
            raw = features.reshape(len(features),-1)
            dim = raw.shape[1]
            if dim >= 128:
                mean,basis,features = awm.pca_fit(raw,keep=128)
            else:
                mean = np.zeros(dim,np.float32)
                basis = np.zeros((dim,128),np.float32)
                basis[:,:dim] = np.eye(dim,dtype=np.float32)
                features = raw @ basis
            projection = out/'student_projection.npz'
            np.savez(projection,mean=mean,basis=basis)
            spec.update(raw_feature_shape=list(spec['feature_shape']),feature_shape=[2,64],
                projection_path=str(projection),projection_sha256=sha(projection))
            spec['representation_sha256'] = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
            features = features.reshape(len(features),2,64)
        data = dict(rows=lib.rows, features=features, rs=lib.rs, episode=lib.episode,
                    task_id=lib.task_id, step=lib.step, action=lib.action[:, :10, :7])
        np.savez(dataout/'library.npz', **data)
        if spec.get('library_features_reference'):
            if sha(spec['library_features_reference'])!=spec['library_features_reference_sha256']:
                raise ValueError('changed frozen PILOT feature reference')
            # Reference is an audited export-parity check after fitting inputs
            # are closed. It cannot train a normalizer, encoder or head.
    fitreads = sorted(reads)
    if spec.get('library_features_reference'):
        with np.load(spec['library_features_reference'],allow_pickle=False) as reference:
            np.testing.assert_array_equal(reference['rows'],lib.rows)
            live=features.reshape(len(features),128)
            delta=float(np.max(np.abs(reference['features']-live)))
            # B1 uses a different CPU convolution batch kernel from astra's
            # batch export check. Library and live serving both use B1 here.
            np.testing.assert_allclose(reference['features'],live,rtol=1e-5,atol=1e-4)
            spec['library_reference_parity']=dict(max_abs=delta,source='astra batch feature export',
                serving='library and online both batch one, exactly equal',atol=1e-4,rtol=1e-5)
    print('ENCODED', size, len(features), flush=True)
    featurelib = FeatureLibrary(STORE, 'pi05_l10', 'bpool_cs', chosen, features=features)
    serving = {k:spec[k] for k in ('factory','kwargs','device','threads','feature_shape','feature_kind',
        'epsilon','cost_basis','cell','completed_epochs','library_only','representation_sha256')}
    serving.update({k:spec[k] for k in ('raw_feature_shape','projection_path','projection_sha256','declared_epochs','cpu_affinity') if k in spec})
    obj = LightKeyKnob(library='bpool_cs', episode_subset=chosen, encoder_spec=serving)
    ctx = api.Context(root=STORE, cell='pi05_l10_cache', seed=0, scratch=dataout/'scratch')
    def identity_pca(X, **unused):
        a = np.asarray(X, np.float32)
        if a.shape[1] != 64:
            raise ValueError('refit tried to consume teacher tokens')
        return np.zeros(64, np.float32), np.eye(64, dtype=np.float32), a.copy()
    diagnostic_audit=dict(queries=0,removed_rows=0,own_episode_donors=0)
    with library_boundary(STORE,libdir), patch.object(fitting, '_Library', return_value=featurelib), patch.object(awm, 'pca_fit', identity_pca), patch.object(awm.AWM,'os_score_all',donor_scores(awm.AWM.os_score_all,diagnostic_audit)):
        fitting.fit_recipe(obj, ctx)
    obj.fit_info.update(feature_kind='light_symmetric', representation_sha256=spec['representation_sha256'],
        pca='identity access to frozen student camera blocks; no teacher-token projection',
        student_fixed_during_donor_LOEO=True, encoder_episode_cross_fitted=False,
        diagnostic_calibration=diagnostic_audit,
        donor_exclusion='before main/fresh median, top-k and kernel distance reference in every LOEO path')
    records = {}
    for level in ('off', 'ir25'):
        path, old = baseline(size, level)
        method = copy.deepcopy(obj)
        method.knob_method = 'off' if level == 'off' else 'periodic'
        method.target = None if level == 'off' else .25
        method.configure(old['method'].knob_settings, 'pi05')
        method.reset(api.EpisodeView('qk-build','',0,0,0,0))
        kwargs = dict(library='bpool_cs', episode_subset=chosen, method=method.knob_method,
                      target=method.target, encoder_spec=serving)
        artifact = out/f'{level}.pkl'
        blob = dict(method=method, registered={}, cell='pi05_l10_cache', spec=SPEC,
                    kwargs=kwargs, fit_s=time.perf_counter()-start)
        artifact.write_bytes(pickle.dumps(blob, protocol=4))
        records[level] = dict(path=str(artifact), sha256=sha(artifact), spec=SPEC, kwargs=kwargs,
                             baseline=str(path), baseline_sha256=sha(path),
                             fixed_knob_settings=method.knob_settings)
    manifest = dict(cell=f'pi05_l10_{size}', encoder_spec=spec, subset=str(subset), subset_sha256=sha(subset),
        parent=str(libdir), selected_row_sha256=hashlib.sha256(lib.rows.tobytes()).hexdigest(),
        selected_image_sha256=[h.hexdigest() for h in image_hash],
        library_features=str(dataout/'library.npz'), library_features_sha256=sha(dataout/'library.npz'),
        sigma=featurelib.action[:, :5,:7].std((0,1)).tolist(),
        rows=len(features), episodes=size, fit_reads=sorted(set(fitreads+obj.fit_info['fit_reads'])),
        fit_info=obj.fit_info, records=records, offline_only=True, closed_loop_launches=0)
    write(out/'manifest.json', manifest)
    write(out/'encoder.json', spec)
    print('BUILT', size, manifest['rows'], flush=True)
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, choices=(50,200), required=True)
    ap.add_argument('--encoder-config', type=Path)
    ap.add_argument('--tag',help='isolated config/refit compatibility build; default is the active pilot artifact')
    args = ap.parse_args()
    build(args.size, args.encoder_config,args.tag)
