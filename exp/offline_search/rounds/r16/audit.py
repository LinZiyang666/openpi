"""Read-only definitions audit and source registration; never touches run outcomes."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from exp.offline_search.closed_loop import devset,stage_overrides
from exp.offline_search.closed_loop.ops.h100 import control
from .prepare import HERE,REPO,CELLS,parts,write_json


def source_list():
    """Use the existing deployment closure rather than guessing dynamic imports."""
    files=control.source_files()
    return {str(relative):dict(source=str(path),sha256=devset.sha(path)) for path,relative in files}


def audit():
    design=json.loads((HERE/'design.json').read_bytes())
    records=design['phase1']+[r['dev'] for r in design['cells'].values()]
    records += list({r['fallback']['root']:r['fallback'] for r in design['cells'].values()}.values())
    if (HERE/'smoke_roots.json').exists():records += json.loads((HERE/'smoke_roots.json').read_bytes())
    fits={};definitions={};counts={}
    for rec in records:
        root=Path(rec['root']);seal=json.loads((root/'ROOT.json').read_bytes())
        assert rec==seal and root.name.startswith('r16_') and seal['intended_host']=='local'
        assert not any(p.is_symlink() for p in root.rglob('*'))
        for name,digest in rec['definitions_sha256'].items():
            assert devset.sha(root/name)==digest;definitions[str(root/name)]=digest
        rows=json.loads((root/'arms.json').read_bytes());assert len(rows)==rec['arm_count']
        for row in rows:
            assert not set(row)&{'server_host','worker_host','fleet'}
            assert row['intended_host']=='local' and row['full_model'] and row['replan_steps']==5
            k=row['r16_k']; assert stage_overrides.miss_steps_from_yaml(row['yaml'],row['model'],k if row['model']=='groot' else None)==k
            if row['model']=='groot':assert int(row['server_env']['GROOT_DENOISING_STEPS'])==k
            assert '--os-policy-tail' in row['plugin_args']
            if row.get('r15_leave_own_episode_out'):assert row['arm'].startswith('r15d_')
            counts[row['model']]=counts.get(row['model'],0)+1
        for path,digest in seal['fit_sha256'].items():assert devset.sha(path)==digest;fits[path]=digest
    for name,count in [('queue_phase1.txt',8),('queue_dev.txt',176),('queue_smoke.txt',4)]:
        lines=(HERE/name).read_text().splitlines();assert len(lines)==count
        assert all(line.startswith('@local /home/weiland/trace_runs/os_closed_loop/r16_') and len(line.split('|')[1].split())==1 for line in lines)
    # AST proof: only the one-step validation exception and lower bound changed.
    before=(HERE/'before/types.py.txt').read_text();after=(REPO/'src/openpi/cache/types.py').read_text()
    tree_before,tree_after=ast.parse(before),ast.parse(after)
    def permitted(tree):
        for node in ast.walk(tree):
            if isinstance(node,ast.ClassDef) and node.name=='DenoiseSchedule':
                node.body=[x for x in node.body if not isinstance(x,ast.FunctionDef) or x.name!='__post_init__']
        tree.body=[x for x in tree.body if not isinstance(x,ast.FunctionDef) or x.name!='groot_n15_schedule']
        return ast.dump(tree,include_attributes=False)
    assert permitted(tree_before)==permitted(tree_after)
    result=dict(PASS=True,roots=len(records),arms=sum(counts.values()),by_model=counts,
                source_definitions_verified=len(definitions),fit_sha256=fits,
                shared_edits=['src/openpi/cache/types.py'],shared_nonvalidator_AST_unchanged=True,
                outcome_reads=0,servers=0,remote_operations=0)
    write_json(HERE/'definition_audit.json',result)
    return definitions,fits


def register():
    if (HERE/'DESIGN_LOCK.json').exists():raise FileExistsError('Source registration already exists')
    definitions,fits=audit()
    sources=source_list()
    runtime_proof=json.loads((HERE/'runtime_proof.json').read_bytes());assert runtime_proof['PASS']
    # Register all deployed code, all local R16 tools, and frozen rule sources.
    source_hashes={r['source']:r['sha256'] for r in sources.values()}
    for p in sorted(HERE.glob('*.py')):source_hashes[str(p)]=devset.sha(p)
    for p in [HERE/'COORDINATOR_DECISIONS.md',HERE/'PROTOCOL.md',HERE/'protocol.json',HERE/'ADAPTER.md',
              HERE/'smoke_commands.sh',HERE/'runtime_proof.json',HERE/'reference_reuse.json',
              HERE/'before/types.py.txt',REPO/'exp/offline_search/rounds/r14/descent_astra/PROTOCOL.md']:
        source_hashes[str(p)]=devset.sha(p)
    write_json(HERE/'DESIGN_LOCK.json',dict(schema='r16.design.lock.v1',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        design_sha256=devset.sha(HERE/'design.json'),source_sha256=source_hashes,outcome_reads=0,
        default_runtime_proof_sha256=devset.sha(HERE/'runtime_proof.json')))
    # Local full-model servers use this repository in place; only one shared file changed.
    local={rel:r for rel,r in sources.items() if rel.startswith(('src/','scripts/','exp/offline_search/','exp/libero_groot/'))}
    worker={rel:dict(**r,destination='/scratch/zixuans8/openpi_trace_h100/'+rel) for rel,r in sources.items()}
    write_json(HERE/'deployment.json',dict(schema='r16.deployment.v1',worker_host='timan107',server_host='local',
        timan107_code_files=worker,local_server_code_files=local,
        local_changed_runtime_files={'src/openpi/cache/types.py':sources['src/openpi/cache/types.py']},
        local_frozen_fit_files=fits,local_root_definition_files=definitions,
        worker_assets='Existing control sync remaps matrices/manifests; local mode keeps full libraries and fits on local server only',
        memory_estimate_mib_per_full_server={'pi05':9000,'groot':8000},
        memory_basis='Existing control.server_spec full-model NEED_MB planning allowances; unmeasured peak; k does not unload weights',
        suggested_concurrent_servers=2,quota_gb=28,real_gpu_measurement_performed=False))
    for host,entries in [('timan107',worker),('local',local)]:
        (HERE/f'deployment_{host}.sha256').write_text(''.join(f'{r["sha256"]}  {rel}\n' for rel,r in sorted(entries.items())))
    print(f'REGISTERED {len(sources)} code files; no outcome reads or deployment')


def deployment_test(tables):
    from .prepare import frozen_tables
    from .analyze import Reader,verify,checked_root
    levels,uniform,freeze=frozen_tables(tables)
    reader=Reader();verify(reader)
    records=reader.json(HERE/'test_roots.json')
    assert {r['cell'] for r in records}==set(CELLS)
    fits,definitions={},{}
    for rec in records:
        root,seal=checked_root(rec,reader)
        table=reader.json(root/'descent_table.json')
        assert table['levels']==levels[rec['cell']] and table['freeze']==freeze
        for path,digest in seal['fit_sha256'].items():
            assert hashlib.sha256(reader.read(path)).hexdigest()==digest;fits[path]=digest
        definitions.update({str(root/name):digest for name,digest in seal['definitions_sha256'].items()})
    reader.stable()
    write_json(HERE/'deployment_test.json',dict(table_freeze=freeze,local_new_fit_files=fits,
        local_root_definition_files=definitions,server_host='local',worker_host='timan107',
        code_manifest='deployment.json (unchanged registered sources)',
        worker_transfers='Existing control sync uses each new root matrix/YAML/manifest; no fit transfer to workers',
        launches=0,outcome_reads=0))
    (HERE/'deployment_test_local.sha256').write_text(''.join(f'{digest}  {path}\n' for path,digest in sorted({**fits,**definitions}.items())))
    print('TEST_DEPLOYMENT_READY 32 roots; no launches or outcome reads')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('verify','register','deployment-test'))
    ap.add_argument('--tables',type=Path,default=HERE/'dev_results');args=ap.parse_args()
    if args.action=='register':register()
    elif args.action=='deployment-test':deployment_test(args.tables)
    else:audit();print('DEFINITIONS_VERIFIED; no outcome reads')


if __name__=='__main__':main()
