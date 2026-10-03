"""One allowed local GPU LOOK latency job, serialized with library encoding."""
import fcntl
import json
import pickle
import time

import numpy as np
import torch

from .encoder import guard_local_gpu, shared_encoder
from .make_arms import HERE,RUN,build
from .prepare import DERIVED,library,write_json
from .validate import online_view


def main():
    with (DERIVED/'.gpu_job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard=guard_local_gpu()
        print(json.dumps({'event':'gpu_guard',**guard}),flush=True)
        enc=shared_encoder('cuda:0')
        reports=[]
        for row in build()[0]:
            path=RUN/'fits'/f'{row["name"]}.pkl'
            if not path.exists(): continue
            with path.open('rb') as f: m=pickle.load(f)['method']
            L=library(row)
            imgs=[np.load(L.dir/'tok'/f'img{i}.npy',mmap_mode='r') for i in (0,1)]
            q=online_view(L,0,imgs)
            for _ in range(5): m.reset(q.episode);m.query(q)
            encode_ms,look_ms=[],[]
            for _ in range(50):
                m.reset(q.episode)
                t=time.perf_counter();r=m.query(q);look_ms.append((time.perf_counter()-t)*1e3)
                encode_ms.append(r.extras['os_clip_encode_ms'])
            def stats(xs): return {'median_ms':float(np.median(xs)),'p95_ms':float(np.percentile(xs,95)),
                                  'min_ms':float(np.min(xs)),'max_ms':float(np.max(xs))}
            reports.append({'arm':row['name'],'repetitions':50,'warmups':5,
                            'two_camera_clip_encode':stats(encode_ms),'full_method_look':stats(look_ms)})
        out={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'PASS':True,
             'scope':'in-process two-camera LOOK; excludes policy stage 1, network, simulator and first model load',
             'precision':'float32, TF32 disabled','ir':'unchanged owner formula; CLIP time reported separately',
             'max_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),**guard,'cells':reports}
        assert out['max_cuda_reserved_bytes']<=3*2**30
        write_json(HERE/'results/latency.json',out)
        print(json.dumps(out),flush=True)


if __name__=='__main__': main()
