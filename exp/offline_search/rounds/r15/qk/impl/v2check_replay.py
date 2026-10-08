"""Prove the published PILOT v2 export works by config/refit alone."""
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import numpy as np
from exp.offline_search.closed_loop import plugin
from . import replay
from .build import HERE, STORE, write
from .encoder import FrozenEncoder, sha


def check(size):
    manifest=json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
    data=dict(np.load(manifest['library_features'],allow_pickle=False))
    method=pickle.loads(Path(manifest['records']['off']['path']).read_bytes())['method']
    encoder=FrozenEncoder(method.light_key_spec)
    parent=STORE/'library/pi05_l10/bpool_cs'
    tokrows=np.load(parent/'tok/rows.npy');inv=np.full(int(tokrows.max())+1,-1,int);inv[tokrows]=np.arange(len(tokrows))
    ims=[np.load(parent/f'tok/img{cam}.npy',mmap_mode='r') for cam in range(2)]
    for t in range(10):
        i=int(np.flatnonzero(data['task_id']==t)[0]);r=int(data['rows'][i])
        look=SimpleNamespace(img0=np.array(ims[0][inv[r]]),img1=np.array(ims[1][inv[r]]),rs=data['rs'][i],task_id=t)
        np.testing.assert_array_equal(encoder.encode(look),data['features'][i])
    with tempfile.TemporaryDirectory(prefix='r15qk-v2-') as tmp:
        conn,s,model,records=replay.connection(plugin,size,'off',Path(tmp),artifact=manifest['records']['off']['path'])
        replay.start(conn,s);row=int(s.method.row_subset[0]);model.poison=True
        conn.infer(replay.observation(s,row,0));conn.infer(replay.observation(s,row,1))
        assert model.towers==model.full==0 and s.step==2
        verdict=s._verdict
        s._verdict=lambda step,conf,ex:dict(verdict(step,conf,ex),hit=False)
        model.poison=False;conn.infer(replay.observation(s,row,2,executed_steps=3))
        assert model.towers==model.full==1 and s.step==3
        model.poison=True;conn.infer(replay.observation(s,row,3))
        assert model.towers==1 and s.step==model.broadcasts==4
    return dict(PASS=True,cell=f'pi05_l10_{size}',candidate=manifest['encoder_spec']['candidate_name'],
        artifact=manifest['records']['off']['path'],artifact_sha256=manifest['records']['off']['sha256'],
        all_task_library_live_exact=True,model_interface='Tensor [N,128], output_index=null',
        stage_calls=dict(HIT=0,MISS=1,blind=0),runtime_code_change=False,
        completed_epochs=method.light_key_spec['completed_epochs'],
        student_sha256=method.light_key_spec['kwargs']['model_sha256'],
        candidate_manifest_sha256=manifest['encoder_spec']['candidate_manifest_sha256'],
        epsilon=method.light_key_spec['epsilon'],status='active frozen final PILOT candidate')


def run():
    report=dict(PASS=True,cells=[check(50),check(200)],runtime_code_change=False)
    write(HERE/'v2check_replay.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':run()
