import copy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from exp.offline_search.rounds.r08.abl.clip import encoder, method
from exp.offline_search.rounds.r08.abl.clip.make_arms import build
from exp.offline_search.harness import api


def test_encoder_matches_openai_reference():
    import open_clip
    import torch
    from PIL import Image
    imgs=[np.random.default_rng(i).integers(0,256,(h,w,3),dtype=np.uint8)
          for i,(h,w) in enumerate(((224,224),(256,256),(129,391)))]
    ref,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained=str(encoder.weights_path()),force_quick_gelu=True)
    ref.eval()
    for im in imgs:
        assert torch.equal(pre(Image.fromarray(im)),encoder.preprocess(im))
    with torch.inference_mode():
        expected=ref.encode_image(torch.stack([pre(Image.fromarray(im)) for im in imgs]),normalize=True).numpy()
    got=encoder.Encoder().encode(imgs)
    assert np.max(np.abs(got-expected))<1e-5
    np.testing.assert_allclose(np.linalg.norm(got,axis=1),1,atol=1e-6)
    # Serving scopes may enable autocast. Retrieval still uses the verified f32 tower.
    enc=encoder.Encoder()
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        cast=enc.encode(imgs)
    assert np.array_equal(got,cast)


@pytest.mark.parametrize('im',[np.zeros((224,224,3),np.float32),np.zeros((3,224,224),np.uint8)])
def test_preprocess_rejects_ambiguous_input(im):
    with pytest.raises(ValueError):
        encoder.preprocess(im)


def test_wrong_weights_fail_closed(tmp_path):
    p=tmp_path/'weights'
    p.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='hash mismatch'):
        encoder.Encoder(path=p)


def test_exact_a_source_rows():
    rows,prov=build()
    assert len(rows)==8 and len({r['name'] for r in rows})==8
    for r,p in zip(rows,prov):
        src=copy.deepcopy(p['source_row'])
        src['name'],src['method']=r['name'],r['method']
        args=src['plugin_args']
        args[args.index('--os-fit-artifact')+1]=r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1]
        assert src==r
        assert '--os-log-inputs' not in r['plugin_args']
        assert r['kwargs']['kref']==(5 if r['kwargs']['lib']=='current' else 8)
        assert method.ClipAWM(**r['kwargs']).k==16


def test_adapter_keeps_request_state_and_records_latency(monkeypatch):
    k=np.arange(1024,dtype=np.float32).reshape(2,512)
    monkeypatch.setattr(encoder,'shared_encoder',lambda: NS(encode=lambda imgs:k.copy()))
    seen=[]
    def query(self,q):
        seen.append(q)
        return api.Result(topk=np.array([0]),scores=np.array([1.]),confidence=1.,
                          action=np.zeros((10,32),np.float32),extras={'still':1.})
    monkeypatch.setattr(method.BlindAWM,'query',query)
    m=method.ClipAWM()
    ep=NS(uid='one')
    q=NS(episode=ep,step=0,img0=np.zeros((224,224,3),np.uint8),img1=np.zeros((224,224,3),np.uint8),
         rs=np.arange(8),hist_rs=np.empty((0,8)),prev_hit=None)
    r=m.query(q)
    assert list(r.extras)[0]=='os_clip_encode_ms' and r.extras['os_clip_encode_ms']>=0
    assert 'still' not in r.extras
    assert seen[-1].rs is q.rs and seen[-1].hist_rs is q.hist_rs
    np.testing.assert_array_equal(seen[-1].key_v0,k[0])
    q.step=1
    r=m.query(q)
    assert r.extras['still']==1.
    assert seen[-1].previous_available
    q.episode=NS(uid='other')
    assert 'still' not in m.query(q).extras
    assert '_encoder' not in vars(m)


def _emit(spec,run):
    from exp.offline_search.closed_loop.ops.emit_arms import main
    main(['--run-root',run,'--spec',spec])


def test_concurrent_emit_preserves_other_arms(tmp_path):
    run=tmp_path/'run'
    specs=[]
    for tag in ('other','clip'):
        p=tmp_path/f'{tag}.json'
        p.write_text(json.dumps([{'name':tag+str(i),'model':'pi05','suite':'l10','mode':'native'} for i in range(3)]))
        specs.append(str(p))
    ctx=multiprocessing.get_context('fork')
    children=[ctx.Process(target=_emit,args=(p,str(run))) for p in specs]
    for c in children: c.start()
    for c in children:
        c.join(30)
        assert c.exitcode==0
    rows=json.loads((run/'arms.json').read_text())
    assert {r['arm'] for r in rows}=={tag+str(i) for tag in ('other','clip') for i in range(3)}
    keep=(run/'config/other0.yaml').read_bytes()
    _emit(specs[1],str(run))
    assert (run/'config/other0.yaml').read_bytes()==keep
    assert len(json.loads((run/'arms.json').read_text()))==6
