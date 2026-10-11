"""Verify sealed native source identities on CPU, without loading weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def audit(host,assets_filter=None):
    q=Path(__file__).resolve().parent;root=q.parents[2];rows=[];seen=set()
    for path in sorted((q/'specs').glob('asset_*.json')):
        job=json.loads(path.read_text());assets=Path(job['assets'])
        if job['host']!=host or str(assets) in seen:continue
        if assets_filter is not None and str(assets)!=str(assets_filter):continue
        if not assets.joinpath('CPU_RECEIPT.json').exists():
            assert job.get('future_checkpoint') and not Path(job['checkpoint']).exists()
            continue
        seen.add(str(assets));manifest=json.loads((assets/'model_manifest.json').read_text())
        load=json.loads((assets/'load.json').read_text());spec=json.loads((assets/'model_spec.json').read_text())
        checked={}
        if job['model']=='pi05':
            for name,expected in manifest['source_digests'].items():
                actual=hashlib.sha256((root/name).read_bytes()).hexdigest();assert actual==expected,name
                checked[name]=actual
        elif job['model']=='groot':
            from exp.r17_integration.models.groot.manifest import digest,file_digest
            pairs=sorted((Path(name).name,file_digest(name)) for name in load['code_files'])
            assert digest(pairs)==manifest['groot']['code_identity']
            checked={name:file_digest(name) for name in load['code_files']}
        elif job['model']=='dp':
            from exp.r17_integration.models.dp.manifest import code_identity,framework_identity
            checked=code_identity()
            for name,actual in checked.items():assert actual==manifest['content_digests'][name],name
            assert framework_identity()==manifest['framework_digests']
        elif job['model']=='fastwam':
            from exp.r17_integration.models.fastwam.loader import LoadSpec,asset_paths
            from exp.r17_integration.models.fastwam.manifest import sha256_file
            for name,source in asset_paths(LoadSpec(**load)).items():
                if name.startswith('code:'):
                    actual=sha256_file(source);assert actual==load['expected_digests'][name],name
                    checked[name]=actual
        else:raise ValueError(job['model'])
        assert checked
        rows.append(dict(asset_id=job['asset_id'],assets=str(assets),status='PASS',
            checkpoint_identity=spec['checkpoint_identity'],norm_identity=spec['norm_identity'],
            code_digest=spec['code_digest'],source_files=len(checked),native_reseal_required=False))
    if assets_filter is not None and not rows:pass
    return dict(status='PASS',host=host,cpu_only=True,gpu_executed=False,weight_load=False,assets=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',choices=('local','h100'),required=True)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    print(json.dumps(audit(a.host)))


if __name__=='__main__':main()
