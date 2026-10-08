"""Emit definitions only. No launch, fleet binding, outcome reads, or refitting."""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import pickle

import yaml
from exp.offline_search.closed_loop import devset, look_recording
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.rounds.r14.descent import prepare as r14
from exp.offline_search.rounds.r15.descent import prepare as r15
from exp.offline_search.rounds.r14.descent.recipe import validate_levels
from .cost import pricing

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TRACE = r15.TRACE
SEED = r15.SEED
ORDER = r15.DEV_ORDER
CELLS = tuple(f'{m}_{s}_{n}_k{k}' for n in (50,100,200,500)
              for m in ('pi05','groot') for s in ('l10','spatial') for k in (1,2))
atomic, write_json = r15.atomic, r15.write_json


def parts(cell):
    model, suite, size, k = cell.split('_')
    return model, suite, int(size), int(k[1:])


def prior(cell):
    return r14 if parts(cell)[2] == 50 else r15


def base_cell(cell):
    return cell.rsplit('_',1)[0]


def source_dev(cell):
    return TRACE / f'r{14 if parts(cell)[2] == 50 else 15}_descent_{base_cell(cell)}'


def reference_dev(cell):
    m,s,n,k = parts(cell)
    return source_dev(cell) if n == 50 else TRACE / f'r15_descent_c10_{m}_{s}'


def read_row(root, level):
    return next(r for r in json.loads((root/'arms.json').read_bytes()) if r['r14_descent_level'] == level)


def reference(root, row):
    """Snapshot design and fit bytes, never run data or summaries."""
    root = Path(root)
    seal = json.loads((root/'ROOT.json').read_bytes())
    for name,digest in seal['definitions_sha256'].items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or devset.sha(path) != digest:
            raise ValueError(f'Changed reference definition: {path}')
    args = row['plugin_args']; fit = args[args.index('--os-fit-artifact')+1]
    if devset.sha(fit) != seal['fit_sha256'][fit]:
        raise ValueError(f'Changed reference fit: {fit}')
    return dict(root=str(root), arm=row['arm'], row=row, seal=seal,
                arms_sha256=devset.sha(root/'arms.json'), seal_sha256=devset.sha(root/'ROOT.json'),
                manifest_sha256=devset.sha(row['manifest']), fit_path=fit, fit_sha256=devset.sha(fit))


def spec_from(root, row, name, manifest, k):
    source = next(s for s in json.loads((root/'arms_in.json').read_bytes()) if s['name'] == row['arm'])
    spec = copy.deepcopy(source)
    spec.update(name=name, manifest=str(manifest))
    spec['yaml_patch'] = {**spec.get('yaml_patch',{}), 'miss':dict(num_steps=k)}
    spec.setdefault('server_env',{})
    if row['model'] == 'groot':
        spec['server_env']['GROOT_DENOISING_STEPS'] = k
    return spec


def seal(root, cell, pool):
    rows = json.loads((root/'arms.json').read_bytes())
    fits = {r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1] for r in rows}
    record = dict(root=str(root),final_root=str(root),cell=cell,arms=[r['arm'] for r in rows],
                  arm_count=len(rows),episodes_per_arm=300 if pool=='B' else 500,
                  init_pool=pool,intended_host='local',staged=False,recording=False,launches=0,
                  definitions_sha256={str(p.relative_to(root)):devset.sha(p) for p in sorted(root.rglob('*'))
                                      if p.is_file() and p.name != 'ROOT.json'},
                  fit_sha256={p:devset.sha(p) for p in sorted(fits)})
    write_json(root/'ROOT.json',record)
    return record


def emit(root, cell, specs, levels, pool, *, source_rows=None):
    if (root/'arms.json').exists():
        raise FileExistsError(root)
    write_json(root/'arms_in.json',specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root,root/'arms_in.json')
    rows = json.loads((root/'arms.json').read_bytes())
    m,s,n,k = parts(cell)
    contracts = {}
    for index,(row,level) in enumerate(zip(rows,levels,strict=True)):
        row.update(dev=pool=='B',init_pool=pool,r10_size=n,r14_descent_level=level,r16_k=k,
                   r16_cell=cell,r15_leave_own_episode_out=pool=='B' and n==500 and level!='c10',
                   intended_host='local',recording=False,rollout_replicate_seed=SEED,knob_coin_seed=0,
                   episodes=300 if pool=='B' else 500,fleet_exclusive=True)
        if 'server_host' in row or 'worker_host' in row or 'fleet' in row:
            raise ValueError('Never bind a fleet inside a root')
        if source_rows:
            old = source_rows[index]
            # Full-step behavior remains the original arm, including all client/plugin flags.
            for field in ('method','kwargs','plugin_args','client_overrides','full_model','cost_ledger'):
                if row.get(field) != old.get(field):
                    raise ValueError(f'Changed lower-layer/execution field {field}')
            a = yaml.safe_load(Path(row['yaml']).read_text())
            b = yaml.safe_load(Path(old['yaml']).read_text())
            a.pop('miss',None); b.pop('miss',None)
            if a != b:
                raise ValueError('YAML changes exceed MISS step override')
    write_json(root/'arms.json',rows)
    for row in rows:
        contract = devset.arm_contract(root,row,load_manifest(row['manifest']))
        if look_recording.validate_arm(row,contract):
            raise ValueError('Invalid emitted arm')
        if contract is not None:
            contracts[row['arm']] = contract
    if pool == 'B':
        write_json(root/'dev_contracts.json',contracts)
    return seal(root,cell,pool)


def queue(path, records):
    atomic(path,''.join(f'@local {r["root"]}|{arm}\n' for r in records for arm in r['arms']).encode())


def build():
    if (HERE/'design.json').exists():
        raise FileExistsError('Design already exists; use verify')
    # Preflight ALL sources before any output root is created.
    test_refs = json.loads((r15.HERE/'c10_proof.json').read_bytes())['test_references']
    dev_refs, full_descent = {},{}
    for cell in CELLS:
        m,s,n,k = parts(cell)
        root = reference_dev(cell)
        dev_refs[cell] = reference(root,read_row(root,'c10'))
        tr = TRACE / f'r{14 if n==50 else 15}_descent_test_{m}_{s}_{n}'
        full_descent[cell] = reference(tr,read_row(tr,'task_descent'))
        for level in ORDER:
            reference(source_dev(cell),read_row(source_dev(cell),level))
    test_refs = {ms:reference(Path(r['root']),read_row(Path(r['root']),'c10')) for ms,r in test_refs.items()}
    jobs = [TRACE/f'r16_phase1_{ms}_k{k}' for ms in test_refs for k in (1,2)]
    jobs += [TRACE/f'r16_dev_{cell}' for cell in CELLS]
    jobs += [TRACE/f'r16_dev_c10_{m}_{s}_{group}_k{k}' for group in ('50','100plus')
             for m in ('pi05','groot') for s in ('l10','spatial') for k in (1,2)]
    if any(p.exists() for p in jobs):
        raise FileExistsError('Fresh R16 roots required')
    phase1, dev, fallback, reuse, cells = [],[],{}, {},{}
    for ms,ref in test_refs.items():
        for k in (1,2):
            cell = f'{ms}_50_k{k}'; root = TRACE/f'r16_phase1_{ms}_k{k}'
            manifest = root/'eval500.json'
            atomic(manifest,Path(ref['row']['manifest']).read_bytes())
            name = f'r16p1_{ms}_k{k}_pure_c10'
            specs = [spec_from(Path(ref['root']),ref['row'],name,manifest,k)]
            phase1.append(emit(root,cell,specs,['c10'],'A',source_rows=[ref['row']]))
    for cell in CELLS:
        m,s,n,k = parts(cell); original = source_dev(cell); ref = dev_refs[cell]
        source_manifest = original/'dev_selection.json'
        manifest_data = json.loads(source_manifest.read_bytes())
        ref_manifest = json.loads(Path(ref['row']['manifest']).read_bytes())
        if manifest_data['pairs'] != ref_manifest['pairs'] or manifest_data['seed'] != ref_manifest['seed']:
            raise ValueError('Full dev C10 identity/seed mismatch')
        c10row = read_row(original,'c10') if n==50 else ref['row']
        for field in ('method','kwargs','plugin_args','client_overrides','full_model','cost_ledger'):
            if c10row.get(field) != ref['row'].get(field):
                raise ValueError('Full dev C10 template mismatch')
        group = '50' if n==50 else '100plus'; key = f'{m}_{s}_{group}_k{k}'
        if key not in fallback:
            froot = TRACE/f'r16_dev_c10_{key}'; fm = froot/'dev_selection.json'
            atomic(fm,Path(ref['row']['manifest']).read_bytes())
            name = f'r16d_{key}_pure_c10'
            fcell = f'{m}_{s}_{50 if n==50 else 100}_k{k}'
            fallback[key] = emit(froot,fcell,[spec_from(Path(ref['root']),ref['row'],name,fm,k)],
                                 ['c10'],'B',source_rows=[ref['row']])
        root = TRACE/f'r16_dev_{cell}'; manifest = root/'dev_selection.json'
        atomic(manifest,source_manifest.read_bytes())
        source_rows = [read_row(original,level) for level in ORDER]
        # R15's frozen LOEO wrapper recognizes the existing r15d_ B UID prefix.
        # Preserve that prefix instead of changing shared runtime identity rules.
        specs = [spec_from(original,row,f'r15d_r16_{cell}_{level}',manifest,k)
                 for row,level in zip(source_rows,ORDER,strict=True)]
        write_json(root/'protocol.json',dict(init_pool='B',order=ORDER,k=k,primary_tolerance_pp=5,
                   first_failure_stop=True,full_step_tolerance_reference=ref['arm'],
                   fallback='k_step_c10',leave_own_episode_out=n==500,fits_include_own_episode=n==500,
                   frozen_protocol=str(r14.R14/'descent_astra/PROTOCOL.md')))
        rec = emit(root,cell,specs,ORDER,'B',source_rows=source_rows); dev.append(rec)
        caps = {l:prior(cell).level_cap(base_cell(cell),l) for l in ('off','c10',*ORDER[:-1])}
        cells[cell] = dict(dev=rec,fallback=fallback[key],full_dev=ref,full_test_descent=full_descent[cell],
                           full_test_c10=test_refs[f'{m}_{s}'],periodic_caps=caps,
                           pairs=manifest_data['pairs'],k=k,model=m,suite=s,size=n)
        reuse[cell] = dict(reusable=True,identity_sha256=devset.sha(manifest),
                          source_manifest_sha256=devset.sha(source_manifest),
                          full_reference_manifest_sha256=ref['manifest_sha256'],
                          same_pairs=True,same_seed=True,same_policy_template=True,
                          reference=ref['arm'],fallback_dev_needed_for_protocol_stitched_sr_ir=True)
    queue(HERE/'queue_phase1.txt',phase1)
    # Shared fallback first, then every cell's five descending levels together.
    queue(HERE/'queue_dev.txt',list(fallback.values())+dev)
    atomic(HERE/'queue_test.txt',b'')
    write_json(HERE/'reference_reuse.json',dict(outcome_reads=0,cells=reuse,
               test_templates_byte_identical=True,lower_layer_edits=0))
    design = dict(schema='r16.design.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
                  seed=SEED,cells=cells,phase1=phase1,full_test_refs=test_refs,
                  costs={m:{str(k):pricing(m,k) for k in (1,2,10 if m=='pi05' else 8)} for m in ('pi05','groot')},
                  outcomes_read=0,server_host='local',worker_host='timan107',
                  fallback_dev_reason='Frozen R14 protocol requires stitched selected SR/IR; 16 shared k-C10 arms',
                  phase1_arm_count=8,dev_descent_arm_count=160,dev_fallback_arm_count=16)
    catalogs=[r14.ROUNDS/'r13/impl/fits.json',r14.ROUNDS/'r14/impl/fits.json',r15.HERE/'fits.json',
              r15.HERE/'c10_proof.json']
    design['source_catalog_sha256']={str(p):devset.sha(p) for p in catalogs}
    write_json(HERE/'design.json',design)
    print('BUILT phase1=8 dev=160 shared_k_c10=16; no launches')


def frozen_tables(directory):
    directory = Path(directory)
    freeze = json.loads((directory/'TABLE_FREEZE.json').read_bytes())
    for name,digest in freeze['sha256'].items():
        if devset.sha(directory/name) != digest:
            raise ValueError(f'Changed frozen table/input: {name}')
    if devset.sha(HERE/'design.json') != freeze['design_sha256']:
        raise ValueError('Changed R16 design')
    levels = json.loads((directory/'levels.json').read_bytes())
    if freeze.get('adaptive'):
        if freeze.get('uniform_tables') is not False or devset.sha(HERE/'CONDUCTOR_AMENDMENT.json')!=freeze['amendment_sha256']:
            raise ValueError('Changed adaptive amendment')
        if (devset.sha(HERE/'CONDUCTOR_LOCK.json')!=freeze['conductor_lock_sha256'] or
            devset.sha(HERE/'PROTOCOL.md')!=freeze['protocol_sha256']):
            raise ValueError('Changed frozen conductor implementation/protocol')
        uniform = {c:None for c in CELLS}
    else:
        uniform = json.loads((directory/'uniform.json').read_bytes())
    if set(levels) != set(CELLS) or set(uniform) != set(CELLS):
        raise ValueError('Need all 32 frozen cells')
    return levels,uniform,freeze


def emit_test(directory):
    levels,uniform,freeze = frozen_tables(directory)  # MUST precede all writes.
    from .analyze import Reader, verify
    reader=Reader();verify(reader);reader.stable()
    design = json.loads((HERE/'design.json').read_bytes())
    records = []
    targets = [TRACE/f'r16_test_{c}' for c in CELLS]
    if any(p.exists() for p in targets):
        raise FileExistsError('Fresh test roots required')
    for cell,root in zip(CELLS,targets,strict=True):
        m,s,n,k = parts(cell); table = validate_levels(levels[cell])
        adapter = prior(cell); blob = adapter.descent_blob(base_cell(cell),table)
        fitpath = HERE/'artifacts'/f'{cell}_task_descent.pkl'
        atomic(fitpath,pickle.dumps(blob,protocol=4))
        fit = dict(path=str(fitpath),sha256=devset.sha(fitpath),spec=blob['spec'],kwargs=blob['kwargs'])
        manifest = root/'eval500.json'; ref = design['cells'][cell]['full_test_c10']
        atomic(manifest,Path(ref['row']['manifest']).read_bytes())
        spec = adapter.arm_spec(base_cell(cell),f'r16t_{cell}_task_descent',fit,manifest)
        spec['yaml_patch'] = dict(miss=dict(num_steps=k))
        if m=='groot': spec['server_env'] = dict(GROOT_DENOISING_STEPS=k)
        write_json(root/'descent_table.json',dict(levels=table,uniform=uniform[cell],k=k,
                   caps=blob['method'].knob_settings['K'],fit=fit,freeze=freeze,
                   off_tasks=[t for t in range(10) if table[str(t)]=='off'],
                   source_artifacts={l:adapter.frozen(base_cell(cell),l,test=True)['sha256'] if n!=50 else
                                     adapter.frozen(base_cell(cell),l)['sha256'] for l in set(table.values())},
                   frozen_at_utc=freeze['frozen_at_utc'],emitted_at_utc=datetime.now(timezone.utc).isoformat()))
        records.append(emit(root,cell,[spec],['task_descent'],'A'))
    write_json(HERE/'test_roots.json',records)
    queue(HERE/'queue_test.txt',records)
    if freeze.get('adaptive'):
        from .conductor import prepare_groups
        prepare_groups('test',tables=directory)
    print('EMITTED 32 frozen per-task test-A arms; measure only')


def build_smokes():
    """Two identities per isolated root, never contaminate the official runs."""
    design = json.loads((HERE/'design.json').read_bytes())
    records = []
    for m in ('pi05','groot'):
        for phase in ('phase1','dev'):
            rec = next(r for r in design['phase1'] if r['cell']==f'{m}_l10_50_k1') if phase=='phase1' else design['cells'][f'{m}_l10_50_k1']['dev']
            original=Path(rec['root']); old=next(r for r in json.loads((original/'arms.json').read_bytes())
                       if r['r14_descent_level']==('c10' if phase=='phase1' else 'ir45'))
            root=TRACE/f'r16_smoke_{phase}_{m}_l10_k1'
            if root.exists():raise FileExistsError(root)
            pool='A' if phase=='phase1' else 'B';source=json.loads(Path(old['manifest']).read_bytes())
            identities = source if pool=='A' else source['pairs']
            identities=sorted(identities)[:2]
            manifest=root/('eval2.json' if pool=='A' else 'dev_selection.json')
            if pool=='A':write_json(manifest,identities)
            else:write_json(manifest,dict(source,pairs=identities,per_task=2,smoke=True))
            name=f'r16smoke_{phase}_{m}_l10_k1'
            spec=spec_from(original,old,name,manifest,1)
            write_json(root/'arms_in.json',[spec])
            with contextlib.redirect_stdout(io.StringIO()):_emit(root,root/'arms_in.json')
            rows=json.loads((root/'arms.json').read_bytes())
            rows[0].update(dev=pool=='B',init_pool=pool,episodes=2,r10_size=50,r14_descent_level=old['r14_descent_level'],
                           r16_k=1,intended_host='local',recording=False,rollout_replicate_seed=SEED,knob_coin_seed=0,
                           fleet_exclusive=True,r16_smoke=True)
            write_json(root/'arms.json',rows)
            if pool=='B':
                contract=devset.arm_contract(root,rows[0],load_manifest(manifest))
                write_json(root/'dev_contracts.json',{name:contract})
            record=seal(root,f'{m}_l10_50_k1',pool)
            record['episodes_per_arm']=2;write_json(root/'ROOT.json',record)
            records.append(record)
    write_json(HERE/'smoke_roots.json',records);queue(HERE/'queue_smoke.txt',records)
    print('BUILT 4 smoke arms x 2 episodes; no launches')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action',required=True)
    sub.add_parser('build')
    sub.add_parser('build-smokes')
    test = sub.add_parser('emit-test'); test.add_argument('--tables',type=Path,default=HERE/'dev_results')
    args = ap.parse_args()
    if args.action=='build': build()
    elif args.action=='build-smokes':build_smokes()
    else: emit_test(args.tables)


if __name__ == '__main__': main()
