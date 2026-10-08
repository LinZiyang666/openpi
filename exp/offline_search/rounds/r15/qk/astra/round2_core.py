"""Round-2 CPU models and symmetric AWM evaluation. No external weights."""
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.offline_search.rounds.r15.qk.astra.prototype import evaluate, task_metric

CONFIGS = {
    'vision_aug': {'tokens':0.,'keypoints':False,'state':False},
    'token_aug': {'tokens':.3,'keypoints':False,'state':False},
    'token_keypoint': {'tokens':.3,'keypoints':True,'state':False},
    'state_drop_aug': {'tokens':0.,'keypoints':False,'state':True},
    'r1_style': {'tokens':0.,'keypoints':False,'state':True,'state_dropout':False,
                 'augmentation':False,'global_normalization':True,'dropout':0.,'weight_decay':.001},
}


def setup():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.sched_getaffinity(0) <= set(range(10,22)) | set(range(54,66))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def image_tensor(data):
    return torch.from_numpy(np.stack([data['img0'],data['img1']],1).transpose(0,1,4,2,3).copy()).float()/255.


def take(data, rows):
    n=len(data['episode'])
    return {k:(v[rows] if hasattr(v,'shape') and v.ndim and len(v)==n and k!='sig' else v) for k,v in data.items()}


def task_normalization(y, tasks, floor):
    means=np.stack([y[tasks==t].mean(0) for t in range(10)])
    scales=np.stack([np.maximum(y[tasks==t].std(0),floor) for t in range(10)])
    return means.astype(np.float32), scales.astype(np.float32)


class Student(nn.Module):
    def __init__(self, config, pm, ps, rm, rs):
        super().__init__()
        self.use_state=bool(config['state']);self.keypoints=bool(config['keypoints'])
        self.state_dropout=bool(config.get('state_dropout',True))
        for k,v in [('pm',pm),('ps',ps),('rm',rm),('rs',rs)]:self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
        self.conv=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,24,3,2,1),nn.SiLU(),nn.Conv2d(24,32,3,2,1),nn.SiLU())
        self.task=nn.Embedding(10,8)
        dim=128 if self.keypoints else 32*7*7*2
        self.fc=nn.Sequential(nn.Linear(dim+8+(8 if self.use_state else 0),256),nn.SiLU(),nn.Dropout(float(config.get('dropout',.1))),nn.Linear(256,128))
        self.spatial_head=nn.Conv2d(32,16,1)
        self.action_head=nn.Linear(128,35)
        axis=torch.linspace(-1,1,7);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('xx',xx.reshape(1,1,49));self.register_buffer('yy',yy.reshape(1,1,49))

    def components(self, images, state, task_id):
        n=images.shape[0]
        z=self.conv(images.reshape(n*2,3,56,56)*2-1)
        if self.keypoints:
            p=torch.softmax(z.flatten(2)*5.,dim=-1)
            features=torch.cat([(p*self.xx).sum(-1),(p*self.yy).sum(-1)],1).reshape(n,128)
        else:features=z.reshape(n,-1)
        pieces=[features,self.task(task_id)]
        if self.use_state:
            r=torch.clamp((state-self.rm)/self.rs,-8,8)
            if self.training and self.state_dropout:r=r*(torch.rand((n,1),device=r.device)>.5)
            pieces.append(r)
        standardized=self.fc(torch.cat(pieces,1))
        prediction=standardized*self.ps[task_id]+self.pm[task_id]
        spatial=self.spatial_head(F.adaptive_avg_pool2d(z,(4,4))).reshape(n,2,16,4,4)
        return prediction,spatial,self.action_head(standardized)

    def forward(self, images, state, task_id):
        return self.components(images,state,task_id)[0]


def encode(model, images, state, task, batch=128):
    model.eval();values=[]
    with torch.inference_mode():
        for lo in range(0,len(images),batch):
            values.append(model(images[lo:lo+batch],state[lo:lo+batch],task[lo:lo+batch]).numpy())
    return np.concatenate(values)


def augmentation(images):
    n=images.shape[0];x=images.reshape(n*2,3,56,56)
    theta=torch.zeros(n*2,2,3)
    scales=1+torch.empty(n*2).uniform_(-.025,.025)
    theta[:,0,0]=scales;theta[:,1,1]=scales
    theta[:,:,2]=torch.empty(n*2,2).uniform_(-2/56,2/56)
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x,grid,padding_mode='border',align_corners=False)
    contrast=torch.empty(n*2,1,1,1).uniform_(.9,1.1)
    gain=torch.empty(n*2,3,1,1).uniform_(.95,1.05)
    bright=torch.empty(n*2,1,1,1).uniform_(-.05,.05)
    return torch.clamp((x-.5)*contrast*gain+.5+bright,0,1).reshape(n,2,3,56,56)


def query_data(tr):
    return {**tr,'label':tr['action'],'label_valid':np.ones(len(tr['action']),bool),
            'uid':np.array([f'library:{e}' for e in tr['episode']])}


def summarize(summary, arrays, dev, reference=None):
    vals=[];tasks={}
    for t in range(10):
        rows=dev['task_id']==t
        groups=[(dev['uid']==e)&dev['label_valid'] for e in np.unique(dev['uid'][rows])]
        groups=[g for g in groups if g.any()]
        a=float(np.mean([arrays['err10'][g].mean() for g in groups]));vals.append(a)
        tasks[str(t)]={'err10':a,'episodes':len(groups)}
        if reference is not None:
            b=float(np.mean([reference['err10'][g].mean() for g in groups]))
            tasks[str(t)].update(reference_err10=b,ratio=a/b)
    return {**summary,'balanced_err10':float(np.mean(vals)),'balanced_by_task':tasks}


def metric_eval(name, library_features, query_features, metric, tr, dev, base, teacher_top=None, exclude=False):
    lx=np.c_[library_features,tr['rs'][:,:8]];qx=np.c_[query_features,dev['rs'][:,:8]]
    s,a=evaluate(name,lx,qx,metric,tr,dev,base,teacher_top,exclude_episode=exclude)
    return summarize(s,a,dev),a


def shuffled_indices(tasks, seed):
    rng=np.random.default_rng(seed);ix=np.arange(len(tasks))
    for t in range(10):
        rr=np.flatnonzero(tasks==t)
        # A nonzero cyclic shift of random order prevents unchanged images.
        order=rng.permutation(rr);ix[order]=np.roll(order,1)
        assert np.all(tasks[ix[rr]]==t)
    return ix


def blindness(model, images, state, tasks, library_features, metric, tr, dev, base, clean_summary, clean_arrays, seeds=(2026100503,2026100504,2026100505), exclude=False):
    trials=[]
    for seed in seeds:
        ix=shuffled_indices(dev['task_id'],seed)
        q=encode(model,images[ix],state,tasks)
        s,a=metric_eval(f'shuffle_{seed}',library_features,q,metric,tr,dev,base,exclude=exclude)
        s['relative_rise']=s['balanced_err10']/clean_summary['balanced_err10']-1
        trials.append(s)
    q=encode(model,torch.zeros_like(images),state,tasks)
    zero,za=metric_eval('zero_images',library_features,q,metric,tr,dev,base,exclude=exclude)
    zero['relative_rise']=zero['balanced_err10']/clean_summary['balanced_err10']-1
    per_task={t:float(np.mean([s['balanced_by_task'][t]['err10'] for s in trials])/clean_summary['balanced_by_task'][t]['err10']-1) for t in clean_summary['balanced_by_task']}
    rise=float(np.mean([s['relative_rise'] for s in trials]));worse=sum(x>0 for x in per_task.values())
    return {'shuffle_trials':trials,'shuffle_mean_relative_rise':rise,'zero':zero,
            'shuffle_per_task_relative_rise':per_task,'tasks_worse_under_shuffle':worse,
            'pass':rise>=.05 and zero['relative_rise']>=.05 and worse>=8,
            'rule':'both balanced RMS10 rises >=5%; >=8/10 tasks worsen under shuffle; state fixed'}
