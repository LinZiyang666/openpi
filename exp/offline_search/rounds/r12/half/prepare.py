"""Bind final camera JSON, recalibrate gap, emit B and stage A; no launches."""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.half_everywhere import R12HalfEverywhere, _SubCache, _TaskJudge, _PiJudge, _GrootJudge
from exp.offline_search.closed_loop.half_everywhere_config import SPEC, SCHEMA, SELECTION, load_table
from exp.offline_search.closed_loop.lookhalf_math import freeze_fit
from exp.offline_search.closed_loop.lookhalf_config import validate_row
from exp.offline_search.closed_loop.look_recording import validate_root
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.rounds.r12.prepare_test_a import TRACE, _replace
from .build import HERE, R12, CELLS, load, dump, write_json, sha, STORE
from .knob import calibrate


def wrap(base, guard, fit, binding, *, knob_method='off', target=None, settings=None):
    kwargs = dict(library=base.library, episode_subset=base.episode_subset,
                  camera_table=binding, knob_method=knob_method, target=target)
    obj = R12HalfEverywhere(**kwargs)
    if obj.table_cell != f'{Path(base.fit_info["parent"]).parent.name}_{base.size}':
        raise ValueError('camera table is for another frozen cell')
    initial = vars(obj).copy()
    vars(obj).update(vars(base))
    vars(obj).update(initial)
    obj.row_subset, obj.size, obj.fit_info = base.row_subset, base.size, dict(base.fit_info)
    deployed = copy.deepcopy(base.inner)
    judge = (_PiJudge if deployed.model == 'pi05' else _GrootJudge).__new__(
        _PiJudge if deployed.model == 'pi05' else _GrootJudge)
    vars(judge).update(vars(deployed))
    cache = _SubCache.__new__(_SubCache)
    vars(cache).update(vars(judge.base))
    cache.camera = obj.camera
    cache.camera_metrics, cache.wrist_heads, cache.scale_sub = fit['camera_metrics'], fit['heads'], fit['scale_sub']
    cache.corrector_enabled = True
    judge.base = cache
    obj.inner = _TaskJudge(judge)
    obj.half_fit, obj.risk_fit = guard.half_fit, guard.risk_fit
    obj.knob_settings = copy.deepcopy(settings if settings is not None else getattr(base, 'knob_settings', None))
    obj.half_bytes = freeze_fit(fit['heads'])+freeze_fit(fit['scale_sub'])+freeze_fit(obj.half_fit)+freeze_fit(obj.risk_fit)
    obj.fit_info.update(fit['fit_info'], camera_table=binding, h2_reused_byte_identical=True,
                        half_knob=settings)
    obj.reset(api.EpisodeView('half-build', '', 0, 0, 0, 0))
    obj.name = f'R12HalfEverywhere_{knob_method}__{base.name}'
    api.check_method_attrs(obj)
    return obj, kwargs


def bind_cell(cell, camera, source, dest, provisional):
    model, suite, size = cell.split('_'); size = int(size)
    guard_source = R12/'guard/artifacts'/cell/'h2.pkl'
    guard = load(guard_source)['method']
    fit_path = HERE/'fits'/cell/'wrist.pkl'
    fit_catalog = json.loads((fit_path.parent/'BUILD.json').read_text())
    if sha(fit_path) != fit_catalog['sha256']:
        raise ValueError('wrist fit SHA256 changed')
    fit = load(fit_path)
    if sha(guard_source) != fit['fit_info']['guard_source_sha256']:
        raise ValueError('H2 source SHA256 changed')
    table_path = dest/'tables'/f'{cell}.json'
    write_json(table_path, dict(schema=SCHEMA, cell=cell, camera=camera, provisional=provisional,
        source=dict(path=str(source), sha256=sha(source))))
    binding = dict(path=str(table_path), sha256=sha(table_path))
    base_path = R12.parent/'r10/recipe/artifacts'/f'r10_recipe_{cell}.pkl'
    base = load(base_path)['method']
    np.testing.assert_array_equal(base.row_subset, guard.row_subset)
    outputs = []
    for knob in ('off', 'periodic') if cell == 'pi05_l10_50' else ('off',):
        target = .32 if knob == 'periodic' else None
        if knob == 'periodic':
            base_path = R12.parent/'r11/knob/artifacts'/f'r11_{cell}_periodic_ir32.pkl'
            base = load(base_path)['method']
            settings = calibrate(model, suite, size, camera, base.library, target)
        else:
            settings = None
        obj, kwargs = wrap(base, guard, fit, binding, knob_method=knob, target=target, settings=settings)
        obj.fit_info.update(full_fit=str(base_path), full_fit_sha256=sha(base_path))
        suffix = cell+('_periodic_ir32' if knob == 'periodic' else '')
        path = dest/'artifacts'/f'{suffix}.pkl'
        dump(path, dict(spec=SPEC, kwargs=kwargs, cell=f'{model}_{suite}_cache', method=obj, registered={}, fit_s=0.))
        outputs.append(dict(cell=cell, suffix=suffix, path=str(path), sha256=sha(path),
            knob=knob, camera_table=binding, setting=settings['setting'] if settings else None))
    return outputs


def emit(root, artifact, base, source, pool, provisional):
    root.mkdir(parents=True, exist_ok=False)
    manifest = root/('dev_selection.json' if pool == 'B' else 'eval500.json')
    manifest.write_bytes(source.read_bytes())
    selected = load_manifest(manifest)
    devset.check_manifest_pool(selected, pool)
    count = 100 if pool == 'B' else 500
    if len(selected['selected']) != count or any(sum(t==task for t,_ in selected['selected']) != count//10 for task in range(10)):
        raise ValueError('incorrect selection count')
    specs, frozen = [], {}
    for off in ([False, True] if pool == 'B' and base['model'] == 'groot' else [False]):
        path = Path(artifact['path']) if not off else R12/'guard/artifacts'/artifact['suffix']/'off.pkl'
        blob = load(path)
        name = root.name+('_off' if off else '_half')
        args = ['--os-root', str(STORE), '--os-no-shadow-native', '--os-tokens', 'off', '--os-blind',
            '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only',
            '--os-fit-artifact', str(path), '--os-lookhalf', 'off' if off else SELECTION]
        if pool == 'B':
            args += ['--os-record-looks', '--os-record-look-images', '56']
        specs.append(dict(name=name, model=base['model'], suite=base['suite_short'], mode='plugin',
            method=blob['spec'], kwargs=blob['kwargs'], plugin_args=args, full_model=True,
            cost_ledger=True, manifest=str(manifest), client_overrides=base['client_overrides']))
        frozen[name] = dict(path=str(path), sha256=sha(path))
    write_json(root/'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root/'arms_in.json')
    rows = json.loads((root/'arms.json').read_text())
    for row in rows:
        row.update(init_pool=pool, server_host='h100' if base['model']=='groot' else 'local')
        if pool == 'B':
            row['dev'] = True
        validate_row(row)
    write_json(root/'arms.json', rows)
    if pool == 'B':
        validate_root(root)
    contracts = [devset.arm_contract(root,row,selected) for row in rows] if pool=='B' else []
    table = load_table(artifact['camera_table'])
    record = dict(root=str(root), final_root=str(TRACE/root.name if pool=='A' else root),
        cell=artifact['cell'], suffix=artifact['suffix'], arms=[r['arm'] for r in rows], init_pool=pool,
        episodes_per_arm=count, server_host=rows[0]['server_host'], provisional=provisional,
        source_selection=str(source), selection_sha256=sha(manifest), selection_exact_copy=True,
        frozen_artifacts=frozen, camera_table=artifact['camera_table'],
        wrist_tasks=[int(t) for t,m in table['camera'].items() if m=='wrist'],
        knob=artifact['knob'], setting=artifact['setting'], contracts=contracts, launches=0)
    write_json(root/'ROOT.json',record)
    return record


def prepare(camera_choice, *, provisional=False, destination=None):
    source = Path(camera_choice).resolve()
    choices = json.loads(source.read_text())
    expected = {f'{m}_{s}_{n}' for m,s,n in CELLS}
    if set(choices) != expected:
        raise ValueError('camera choice must include exactly the five frozen cells')
    if not provisional and any(r.get('sources') != ['screen','camdev'] for r in choices.values()):
        raise ValueError('final emission requires screen + camdev; use --provisional for screen-only')
    cameras = {}
    for cell,row in choices.items():
        values = row.get('camera')
        if not isinstance(values,list) or len(values)!=10 or any(m not in ('wrist','both') for m in values):
            raise ValueError('invalid camera choices')
        cameras[cell] = dict(zip(map(str,range(10)),values))
    dest = Path(destination).resolve() if destination else HERE/('provisional' if provisional else 'final')
    # Final pilots live in half/pilots; binding and dry roots have separate homes.
    roots_base = dest if provisional or destination else HERE
    if (roots_base/'EMISSION.json').exists():
        raise FileExistsError(f'binding already emitted: {roots_base}; use a fresh --destination')
    suffixes = [c+s for c in sorted(expected) if 'wrist' in cameras[c].values()
                for s in (('', '_periodic_ir32') if c=='pi05_l10_50' else ('',))]
    for suffix in suffixes:
        for group,prefix in (('pilots','r12_half'),('test_a_staged','r12_test_half')):
            root = roots_base/group/f'{prefix}_{suffix}'
            if root.exists():
                raise FileExistsError(f'fresh emission required: {root}')
    templates = [TRACE/f'r11_knob_{i}/eval500.json' for i in range(1,5)]
    if len({p.read_bytes() for p in templates}) != 1:
        raise ValueError('R11 A templates differ')
    test_rows = [row for i in range(1,5) for row in json.loads((TRACE/f'r11_knob_{i}/arms.json').read_text())]
    B,A,artifacts,skipped = [],[],[],[]
    for model,suite,size in CELLS:
        cell = f'{model}_{suite}_{size}'
        bound = bind_cell(cell,cameras[cell],source,dest,provisional)
        artifacts.extend(bound)
        if 'wrist' not in cameras[cell].values():
            skipped.append(dict(cell=cell,reason='no wrist tasks; B and A skipped'))
            continue
        base = next(r for r in json.loads((TRACE/f'r11_dev_k10_size{size}/arms.json').read_text())
                    if r['model']==model and r['suite_short']==suite)
        test_base = next(r for r in test_rows if r['model']==model and r['suite_short']==suite)
        for artifact in bound:
            B.append(emit(roots_base/'pilots'/f'r12_half_{artifact["suffix"]}',artifact,base,Path(base['manifest']),'B',provisional))
            A.append(emit(roots_base/'test_a_staged'/f'r12_test_half_{artifact["suffix"]}',artifact,test_base,templates[0],'A',provisional))
    for name,data in (('pilot_roots',B),('test_a_roots',A),('artifacts',artifacts),('skipped',skipped)):
        write_json(roots_base/f'{name}.json',data)
    result = dict(provisional=provisional,camera_choice=dict(path=str(source),sha256=sha(source)),
                  B=B,A=A,skipped=skipped,launches=0)
    write_json(roots_base/'EMISSION.json',result)
    print(json.dumps(dict(provisional=provisional,B=len(B),A=len(A),skipped=skipped),indent=2))
    return result


def install_test_a(destination=TRACE):
    records = json.loads((HERE/'test_a_roots.json').read_text())
    pairs = [(Path(r['root']),Path(destination)/Path(r['final_root']).name) for r in records]
    for record,(source,target) in zip(records,pairs):
        if record['provisional'] or target.exists():
            raise ValueError('A installation requires final definitions and fresh destination')
        load_table(record['camera_table'])
        for fit in record['frozen_artifacts'].values():
            if sha(fit['path']) != fit['sha256']:
                raise ValueError('frozen fit changed')
    for source,target in pairs:
        target.mkdir(parents=True)
        shutil.copyfile(source/'eval500.json',target/'eval500.json')
        for name in ('arms.json','arms_in.json','ROOT.json'):
            data = _replace(json.loads((source/name).read_text()),source,target)
            if name == 'ROOT.json':
                data['root'] = data['final_root'] = str(target)
            write_json(target/name,data)
        shutil.copytree(source/'config',target/'config')
    return [str(t) for _,t in pairs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera-choice',type=Path)
    ap.add_argument('--provisional',action='store_true')
    ap.add_argument('--destination',type=Path)
    ap.add_argument('--install-test-a',action='store_true')
    args = ap.parse_args()
    if args.install_test_a:
        print(json.dumps(install_test_a(args.destination or TRACE)))
    elif args.camera_choice:
        prepare(args.camera_choice,provisional=args.provisional,destination=args.destination)
    else:
        ap.error('--camera-choice required')


if __name__=='__main__':
    main()
