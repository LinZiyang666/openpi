"""SHA-attested B-only light feature shards, without tokens or shadow work."""
import json
import numpy as np
from exp.offline_search.closed_loop.look_recording import LookWriter, retrieval_snapshot, json_value
from exp.offline_search.closed_loop import devset


class LightLookWriter(LookWriter):
    def __init__(self, opts, model):
        if opts.os_record_look_tokens or opts.os_record_look_shadow or opts.os_record_look_cache_proposal:
            raise ValueError('light recording accepts features/images and served MISS labels; no tokens or extra shadows')
        super().__init__(opts,model)
        self.data['fields'].pop('key_v0')
        self.data['fields'].pop('key_v1')
        self.data['fields'].pop('tokens')
        self.data['fields'].update(light_features='frozen student [2,64]; agentview,wrist blocks',
            representation_sha256='frozen representation identity',policy_label='served full policy on MISS only')
        self._flush()

    def capture(self,s,ctx,topk,scores,extras):
        out=retrieval_snapshot(s.method,topk,scores,extras)
        out.update(light_features=np.array(ctx.query_keys['light_features'],np.float32,copy=True),
            rs=np.array(ctx.query_keys['robot_state'],np.float32,copy=True),
            representation_sha256=np.array(s.method.light_key_spec['representation_sha256']),
            key_kind=np.array('light_symmetric'))
        if self.opts.os_record_look_raw_distance:
            from .runtime import LightQueryView
            from .recipe import FeatureGeometry
            out['raw_distance']=np.array(np.min(s.method.inner.base._dist(
                FeatureGeometry(LightQueryView(s,s.step,s.ep.task_id,s.ep)))[7]),np.float64)
        size=self.opts.os_record_look_images
        if size:
            for cam,field in enumerate(('observation/image','observation/wrist_image')):
                im=np.asarray(s.cur_obs[field],np.uint8)
                y=np.minimum((np.arange(size)*im.shape[0]/size).astype(int),im.shape[0]-1)
                x=np.minimum((np.arange(size)*im.shape[1]/size).astype(int),im.shape[1]-1)
                out[f'img{cam}']=im[y[:,None],x[None,:]].copy()
        return out

    def write(self,s,d,wire,adapter):
        if not d.get('vision',True):
            return
        snap=d['look_snapshot']; step=d['step']; hit=bool(d['hit'])
        arrays={k:v for k,v in snap.items() if isinstance(v,np.ndarray)}
        meta=dict(schema=self.data['schema'],dev=True,init_pool='B',arm=self.contract['arm'],
            model=self.contract['model'],suite=self.contract['suite'],**s.ep_meta,
            step=step,ctx_step=d['ctx_step'],library=d['lib'],key_kind='light_symmetric',
            representation_sha256=s.method.light_key_spec['representation_sha256'],
            label_source='unlabelled_hit' if hit else 'served_policy',extra_shadow_work=False)
        arrays.update(step=np.array(step),task_id=np.array(s.ep.task_id),init=np.array(s.ep.init),
            raw_state=s.b_raw.a[step].copy(),action=s.b_aex.a[step].copy(),
            wire_actions=np.array(wire['actions'],copy=True),hit=np.array(hit),label_valid=np.array(not hit),
            light_encoder_ms=np.array(d['light_encoder_ms']),
            qk_work=np.array(json.dumps(dict(Q=1,T=int(not hit),M=int(not hit),N=1))),
            meta=np.array(json.dumps(json_value(meta),sort_keys=True)),
            guard_state=np.array(json.dumps(json_value(snap['guard_state']),sort_keys=True)))
        if not hit:
            arrays['policy_label']=s.b_aex.a[step].copy()
        with self.lock:
            name=f'look_{self.id}_{len(self.data["shards"]):07d}.npz'
            path=self.directory/name;temp=path.with_suffix('.tmp')
            with temp.open('xb') as f:
                np.savez_compressed(f,**arrays)
            temp.replace(path)
            self.data['shards'].append(dict(file=name,sha256=devset.sha(path),uid=meta['uid'],
                attempt=meta['attempt'],step=step,task_id=meta['task_id'],init=meta['init']))
            self._flush()
