"""CPU structural/lifecycle tests for server-owned overrides, using cheap towers."""
import dataclasses,json,pathlib,tempfile,types
import torch
from exp.offline_search.rounds.r04.k3_cost.dev import stage_overrides as co
from openpi.models_pytorch.pi0_pytorch import Stage1Output,PI0Pytorch
from openpi.serving import stage_io
HERE=pathlib.Path(__file__).resolve().parent

class Tower:
    def __init__(self):
        self.paligemma=types.SimpleNamespace(vision_tower=torch.nn.Linear(1,1),multi_modal_projector=torch.nn.Linear(1,1))
        self.calls=[]
    def embed_image(self,x):
        self.calls.append(float(x.mean()))
        return x.mean((1,2,3))[:,None,None].expand(-1,256,4).clone()
    def embed_language_tokens(self,x):return x[:,:,None].expand(-1,-1,4).float()

class Model:
    pi05=True
    def __init__(self):self.paligemma_with_expert=Tower()
    def _preprocess_observation(self,obs,train=False):
        return list(obs.images.values()),list(obs.image_masks.values()),obs.tokens,obs.lang_mask,obs.state
    def _prepare_attention_masks_4d(self,m):return torch.where(m[:,None],0.,torch.finfo(torch.float32).min)


def observation(v,b=1):
    names=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']
    return types.SimpleNamespace(images={k:torch.full((b,3,4,4),x) for k,x in zip(names,[v,2*v,-1.])},
        image_masks={k:torch.full((b,),i<2) for i,k in enumerate(names)},tokens=torch.ones((b,2)),
        lang_mask=torch.tensor([[True,False]]).expand(b,-1),state=torch.zeros((b,32)))


def main():
    checks=[]
    orig=PI0Pytorch.run_stage1
    assert co.install_pi05() is None and PI0Pytorch.run_stage1 is orig;checks.append('absent options leave methods identical')
    try:co.install_pi05(pack_prefix=True)
    except ValueError:pass
    else:raise AssertionError('packing was deployed')
    checks.append('packing refused after failed numerical parity')
    m=Model();d=co.Pi05Override();d.stage1(m,observation(.1));d.stage1(m,observation(.2))
    assert m.paligemma_with_expert.calls.count(-1.)==1;checks.append('dummy tower once across two decisions')
    m.paligemma_with_expert.paligemma.vision_tower.weight.data = m.paligemma_with_expert.paligemma.vision_tower.weight.data.clone()
    d.stage1(m,observation(.2));assert m.paligemma_with_expert.calls.count(-1.)==2
    checks.append('weight identity change invalidates cache')
    bad=observation(.1);bad.images['right_wrist_0_rgb'].fill_(0.)
    try:d.stage1(m,bad)
    except ValueError:pass
    else:raise AssertionError('nonconstant dummy accepted')
    bad=observation(.1);bad.image_masks['right_wrist_0_rgb'].fill_(True)
    try:d.stage1(m,bad)
    except ValueError:pass
    else:raise AssertionError('unmasked dummy accepted')
    checks.append('reject changed dummy pixels and mask')
    w=co.Pi05Override('wrist_only').install();m=Model()
    # Two decisions are in flight before either MISS completes: no global stash.
    a=w.stage1(m,observation(.1));b=w.stage1(m,observation(.9))
    ca=w.complete(m,a);cb=w.complete(m,b)
    assert torch.allclose(ca.prefix_embs[:,:256],torch.full((1,256,4),.1))
    assert torch.allclose(cb.prefix_embs[:,:256],torch.full((1,256,4),.9))
    assert a.prefix_embs[:,:256].count_nonzero()==0;checks.append('HIT placeholder immutable and concurrent MISS images isolated')
    bat=stage_io.stack_stage1_output([b,a]);shards=stage_io.split_stage1_output(bat,2)
    for out,ref in zip(shards,[cb,ca]):
        assert torch.equal(w.complete(m,out.to('cpu')).prefix_embs,ref.prefix_embs)
    checks.append('split/reordered rebatch/to preserve deferred image')
    completed=stage_io.stack_stage1_output([ca,cb]);assert not hasattr(completed,'deferred_base')
    try:stage_io.stack_stage1_output([a,cb])
    except ValueError:pass
    else:raise AssertionError('mixed pending/completed accepted')
    checks.append('completed stage3 rebatch supported, mixed status rejected')
    x=w.pack(ca);assert x.prefix_embs.shape[1]==514 and torch.equal(x.prefix_position_ids[:,:512],ca.prefix_position_ids[:,:512])
    checks.append('packing preserves kept positions and drops 256 dummy tokens')
    class Runtime:
        def emit(self,row):self.last=row
    plugin=types.SimpleNamespace(PluginRuntime=Runtime)
    co.install_startup_hook(plugin,stage1_mode='wrist_only',miss_steps=2)
    rt=Runtime();rt.emit({'ev':'startup'});assert rt.last['miss_steps']==2 and rt.last['stage1_mode']=='wrist_only'
    row={'ev':'dec'};rt.emit(row);assert rt.last is row;checks.append('startup hook fields, decision row identity unchanged')
    with tempfile.TemporaryDirectory(dir=HERE/'results') as td:
        f=pathlib.Path(td)/'a.yaml';f.write_text('{}')
        assert co.miss_steps_from_yaml(f,'pi05')==10 and co.miss_steps_from_yaml(f,'groot',8)==8
        f.write_text('miss: {num_steps: 2}')
        assert co.miss_steps_from_yaml(f,'pi05')==2 and co.miss_steps_from_yaml(f,'groot',2)==2
        try:co.miss_steps_from_yaml(f,'groot',8)
        except ValueError:pass
        else:raise AssertionError('GR00T mismatch accepted')
    checks.append('MISS defaults K10/K8, K2 overrides, GR00T mismatch refusal')
    (HERE/'results/override_checks.json').write_text(json.dumps({'passed':len(checks),'checks':checks},indent=2))
    print(json.dumps({'passed':len(checks),'checks':checks},indent=2))
if __name__=='__main__':main()
