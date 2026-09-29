"""Validate the eight exact source spec copies, emit/parse, and record final fit hashes and resources."""
import copy
import hashlib
import json
import pickle
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE,RUN,artifact
import openpi.cache.config as cc


def main():
    rows=json.loads((HERE/'arms_pca.json').read_text())
    provenance={r['arm']:r for r in json.loads((HERE/'results/pca/provenance.json').read_text())}
    outdir=RUN/'pca_validation'; outdir.mkdir(exist_ok=False)
    for r in rows:
        src=provenance[r['name']]['source_row']; restored=copy.deepcopy(r)
        restored['name'],restored['method']=src['name'],src['method']
        restored['kwargs'].pop('pooling_grid')
        restored['plugin_args'][restored['plugin_args'].index('--os-fit-artifact')+1]=artifact(src)
        assert restored==src
    resolved=json.loads(json.dumps(rows).replace('<RUN>',str(RUN)))
    spec=outdir/'spec.json';spec.write_text(json.dumps(resolved,indent=1)+'\n')
    emit(['--run-root',str(outdir),'--spec',str(spec)])
    emitted=json.loads((outdir/'arms.json').read_text()); fits=[]
    for r in emitted:
        cc.load_cache_config(r['yaml'])
        opts,rest=plugin.parse_cli(['--os-method',r['method'],'--os-kwargs',json.dumps(r['kwargs']),
            '--os-cell',r['cell'],'--os-log-dir',str(outdir/'logs'),*r['plugin_args']])
        assert not rest and opts.os_tokens=='on' and opts.os_blind and opts.judge is None
        path=Path(opts.os_fit_artifact)
        with path.open('rb') as f: blob=pickle.load(f)
        assert {k:blob[k] for k in ('spec','kwargs','cell')}==dict(spec=opts.os_method,kwargs=opts.kwargs,cell=opts.os_cell)
        m=blob['method'];assert m.pooling_grid==16 and m.B0T.shape==m.B1T.shape==(64,524288)
        assert m.k==16 and m.budget==1 and m.serving=='anchor_tail'
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1<<24),b''):h.update(b)
        fits.append(dict(arm=r['arm'],path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),
                         fit_seconds=blob['fit_s'],pca=m.token_pca_info,projection_bytes_per_camera=m.B0T.nbytes))
    assert len(fits)==8
    (HERE/'results/pca/validation.json').write_text(json.dumps(dict(PASS=True,source_fields_exact=True,arms=8,artifacts=fits),indent=1)+'\n')
    (HERE/'results/pca/fit_sha256.txt').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in fits))
    print(json.dumps(dict(PASS=True,arms=8,fit_bytes=sum(x['bytes'] for x in fits))),flush=True)


if __name__=='__main__':main()
