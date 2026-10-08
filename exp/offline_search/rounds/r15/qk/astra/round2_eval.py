"""Final B-only assessment, image-ablation audit and stable CPU student export."""
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import torch
from torch import nn
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump,sha
from exp.offline_search.rounds.r15.qk.astra.neural import TinySpatial
from exp.offline_search.rounds.r15.qk.astra.prototype import task_metric
from exp.offline_search.rounds.r15.qk.astra.round2_core import (Student,setup,image_tensor,encode,query_data,metric_eval,blindness,summarize)


class Round1(nn.Module):
    def __init__(self, model):
        super().__init__();self.model=model
    def forward(self, images, state, task_id):
        return self.model(images,state,task_id)[0]


def load_model(size,name):
    cell=f'pi05_l10_{size}'
    if name=='r1_action':
        old=HERE/cell;ck=torch.load(old/'cnn_action_cpu.pt',map_location='cpu',weights_only=True);sd=ck['state_dict']
        base=TinySpatial(sd['pm'],sd['ps'],sd['rm'],sd['rs']);base.load_state_dict(sd);model=Round1(base)
        features=np.load(old/'cnn_action_cpu_features.npz')['library']
        metric=pickle.load((old/'cnn_action_cpu_symmetric_metric.pkl').open('rb'))
    else:
        root=HERE/'round2'/cell/name;ck=torch.load(root/'final.pt',map_location='cpu',weights_only=True);sd=ck['state_dict']
        model=Student(ck['config'],sd['pm'],sd['ps'],sd['rm'],sd['rs']);model.load_state_dict(sd)
        features=np.load(root/'library_features.npz')['features'];metric=pickle.load((root/'metric.pkl').open('rb'))
    return model.eval(),features,metric,ck


def uncertainty(arrays,reference,dev):
    values=[];rng=np.random.default_rng(20261005)
    for task in range(10):
        groups=[dev['uid']==u for u in np.unique(dev['uid'][dev['task_id']==task])]
        values.append(np.array([[arrays['err10'][g].mean(),reference['err10'][g].mean()] for g in groups]))
    draws=np.mean([v[rng.integers(len(v),size=(2000,len(v)))].mean(1) for v in values],axis=0)
    return np.quantile(draws[:,0]/draws[:,1]-1,[.025,.975]).tolist()


def run(size,name):
    setup();cell=f'pi05_l10_{size}';root=HERE/'round2'/cell/name;root.mkdir(parents=True,exist_ok=True)
    tr=dict(np.load(DERIVED/cell/'train.npz'));_,base,_,_,_=load_cell('l10',size)
    model,features,metric,ck=load_model(size,name)
    # The library pseudoquery assessment precedes opening any B-dev data here.
    ld=query_data(tr);lr=np.load(HERE/cell/'loeo_deployed_key_geometry.npz')
    ls,la=metric_eval(name,features,features,metric,tr,ld,base,lr['top'],exclude=True)
    ls=summarize(ls,la,ld,lr);np.savez(root/'loeo_eval.npz',**la)
    assert not np.any(tr['episode'][la['top']]==tr['episode'][:,None])
    library_reference=float(np.mean([np.mean([lr['err10'][tr['episode']==e].mean() for e in np.unique(tr['episode'][tr['task_id']==t])]) for t in range(10)]))
    ls['balanced_ratio']=ls['balanced_err10']/library_reference
    dp=DERIVED/'dev_pi05_l10_r12_devlooks_50L';mask=np.load(dp/f'unseen_{size}.npy')
    with np.load(dp/'dev.npz') as source:
        dev={k:source[k][mask] for k in ('img0','img1','rs','task_id','step','uid','label','label_valid','init')}
    assert dev['label_valid'].all()
    im=image_tensor(dev);rs=torch.from_numpy(dev['rs'][:,:8].copy());tasks=torch.from_numpy(dev['task_id'].astype(np.int64))
    qp=encode(model,im,rs,tasks);ref=np.load(HERE/cell/'eval_reference.npz')
    s,a=metric_eval(name,features,qp,metric,tr,dev,base,ref['top']);s=summarize(s,a,dev,ref)
    clean=s['balanced_err10'];reference=float(np.mean([x['reference_err10'] for x in s['balanced_by_task'].values()]))
    s['balanced_ratio']=clean/reference;s['balanced_relative_excess_ci95']=uncertainty(a,ref,dev)
    progress=tr['progress'][a['top'][:,0]];rp=tr['progress'][ref['top'][:,0]];same=dev['uid'][1:]==dev['uid'][:-1]
    s['phase_wrong_third_vs_teacher']=float(np.mean(np.abs(progress-rp)>1/3))
    s['phase_large_backwards']=float(np.mean((progress[1:]-progress[:-1]<-.1)[same]))
    s['reference_phase_large_backwards']=float(np.mean((rp[1:]-rp[:-1]<-.1)[same]))
    audit=blindness(model,im,rs,tasks,features,metric,tr,dev,base,s,a)
    np.savez(root/'bdev_eval.npz',**a);np.savez(root/'query_features.npz',features=qp)
    result={'cell':cell,'name':name,'comparison':'symmetric student + same library AWM and early metric + 16-NN kernel',
        'episodes':len(np.unique(dev['uid'])),'looks':len(dev['uid']),'library':ls,'b_dev':s,'blindness':audit,
        'cpu_only':True,'gpu_used':False,'current_look_teacher_tokens_used':False,'training_used_bdev':False}
    dump(root/'assessment.json',result)
    print('ASSESSMENT',size,name,'LOEO',ls['balanced_ratio'],'Bdev',s['balanced_ratio'],'top16',s['top16_overlap'],'shuffle',audit['shuffle_mean_relative_rise'],'zero',audit['zero']['relative_rise'],'blind_pass',audit['pass'],flush=True)


def export(size,name):
    setup();cell=f'pi05_l10_{size}';root=HERE/'round2'/cell/name
    out=HERE/'round2'/'pilot'/cell;out.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'assessment.json').read_text());assert report['blindness']['pass']
    model,features,metric,ck=load_model(size,name);tr=dict(np.load(DERIVED/cell/'train.npz'))
    _,base,lib,rows,provenance=load_cell('l10',size)
    scripted=torch.jit.freeze(torch.jit.script(model));deltas=[]
    for task in range(10):
        ids=np.flatnonzero(tr['task_id']==task)[[0,len(np.flatnonzero(tr['task_id']==task))//2,-1]]
        data={k:v[ids] for k,v in tr.items() if k in ('img0','img1')};im=image_tensor(data)
        rs=torch.tensor(tr['rs'][ids,:8].copy());tt=torch.full((3,),task,dtype=torch.long)
        with torch.inference_mode():
            for image,state in ((im,rs),(torch.zeros_like(im),rs),(im,rs+.1)):
                a=model(image,state,tt);b=scripted(image,state,tt)
                assert torch.allclose(a,b,rtol=1e-5,atol=1e-4)
                deltas.append(float((a-b).abs().max()))
    scripted.save(str(out/'student.ts'));torch.save(ck,out/'student.pt')
    np.savez(out/'library_features.npz',features=features,rows=rows)
    with (out/'metric.pkl').open('wb') as f:pickle.dump(metric,f)
    im=image_tensor(tr);rs=torch.tensor(tr['rs'][:,:8].copy());tt=torch.tensor(tr['task_id'].astype(np.int64))
    parity=encode(scripted,im,rs,tt);delta=float(np.max(np.abs(parity-features)))
    assert np.allclose(parity,features,rtol=1e-5,atol=1e-4)
    timings=[]
    for threads in (1,4):
        torch.set_num_threads(threads);samples=[]
        for task in range(10):
            r=int(np.flatnonzero(tr['task_id']==task)[0]);row=int(rows[r])
            raw=np.stack([np.asarray(np.load(lib/f'tok/img{c}.npy',mmap_mode='r')[row]) for c in range(2)])
            state=torch.from_numpy(tr['rs'][r:r+1,:8].copy());tid=torch.tensor([task]);wf=metric[task][0]['wf'];shift=metric[task][0]['shift']
            def call():
                image=torch.from_numpy(raw[:,::4,::4].transpose(0,3,1,2).copy()).float()[None]/255.
                p=scripted(image,state,tid).numpy()
                return np.c_[p,state.numpy()]@wf-shift
            with torch.inference_mode():
                for _ in range(20):call()
                for _ in range(40):
                    start=time.perf_counter_ns();code=call();samples.append((time.perf_counter_ns()-start)/1e6)
            assert np.isfinite(code).all()
        timings.append({'cpu_threads':threads,'n':len(samples),'median_ms':float(np.median(samples)),
            'q1_ms':float(np.quantile(samples,.25)),'q3_ms':float(np.quantile(samples,.75)),
            'p95_ms':float(np.quantile(samples,.95)),'samples_ms':samples})
    latency={'input':'both raw224 RGB -> nearest56 -> float -> frozen student -> task main AWM CPU code; no common synthesis',
             'cpu_only':True,'tasks':list(range(10)),'timings':timings}
    dump(out/'latency.json',latency)
    training_path=(HERE/cell/'cnn_action_cpu_train.json') if name=='r1_action' else root/'final_train.json'
    spec={'factory':'exp.offline_search.rounds.r15.qk.impl.encoder:CNNStudent',
          'kwargs':{'model_path':str(out/'student.ts'),'model_sha256':sha(out/'student.ts'),'output_index':None},
          'device':'cpu','threads':1,'cpu_affinity':list(range(10,22))+list(range(54,66)),
          'feature_shape':[2,64],'feature_kind':'light_symmetric','cell':cell,
          'completed_epochs':ck['epochs'],'declared_epochs':ck['epochs'],'library_only':True,
          'candidate_name':name,
          'library_features_reference':str(out/'library_features.npz'),
          'library_features_reference_sha256':sha(out/'library_features.npz'),
          'epsilon':.152*timings[0]['median_ms']/10.26,
          'cost_basis':'round2 local one-thread raw-image-to-AWM-code median; owner pi05 graph tower10.26ms and cv=.152',
          'source_checkpoint':str(out/'student.pt'),'source_checkpoint_sha256':sha(out/'student.pt'),
          'source_training':str(training_path),'source_training_sha256':sha(training_path),
          'training_provenance':provenance,'input_recipe':'both224 uint8 RGB, nearest4*j to56, CHW /255; normalized rs8; task0..9',
          'export_parity_max_abs':max(deltas),'all_library_export_parity_max_abs':delta}
    spec['representation_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    dump(out/'encoder_config.json',spec)
    manifest={'format':'r15_qk_student_v2','cell':cell,'name':name,'epochs':ck['epochs'],
        'module_signature':'forward(images: float32[N,2,3,56,56], state: float32[N,8], task_id: int64[N]) -> float32[N,128]',
        'image_preprocessing':'agentview,wrist RGB uint8[224,224,3], nearest indices4*j, CHW, /255; no teacher tokens',
        'state_preprocessing':'existing normalized valid rs[:8]; encoder normalizers are internal; downstream AWM appends exact state',
        'features':'128 light features, not teacher token keys; no further teacher PCA; concat exact rs[:8] before AWM',
        'task_range':[0,9],'cpu_only':True,'provenance':provenance,'rows':len(rows),
        'parent_row_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),
        'all_task_perturbation_parity_max_abs':max(deltas),'all_library_export_parity_max_abs':delta,
        'blindness_pass':report['blindness']['pass'],'blindness_assessment_sha256':sha(root/'assessment.json'),
        'offline_bdev_ratio':report['b_dev']['balanced_ratio'],
        'gates_sha256':sha(HERE/'GATES.md'),'full_runtime_refit':'sol must fit layers1-3 from same library; metric.pkl is symmetric exploratory geometry',
        'files':{f:sha(out/f) for f in ('student.ts','student.pt','library_features.npz','metric.pkl','latency.json','encoder_config.json')}}
    dump(out/'manifest.json',manifest)
    print('EXPORTED',cell,name,[(x['cpu_threads'],x['median_ms']) for x in timings],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,required=True);ap.add_argument('--name',required=True);ap.add_argument('--export',action='store_true');a=ap.parse_args()
    if a.export:export(a.size,a.name)
    else:run(a.size,a.name)
