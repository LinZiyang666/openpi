"""Tiny real simulator capture using CPU OSMesa only; no policy or GPU calls."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',
        LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LP_NUM_THREADS='1')
    sys.path.insert(0,a.runtime)
    from exp.metaworld import env as kernel
    from concurrent_clients import decision_seed
    benchmark=json.loads((Path(__file__).parent/'concurrent/bundles/pi05/assets/benchmark.json').read_text())
    from capture import Recorder
    from OpenGL import GL
    import numpy as np
    task=benchmark['tasks'][0][0]
    prompt=benchmark['tasks'][0][1]
    env=kernel.make_env(task,0,7)
    try:
        obs=kernel.reset_and_settle(env)
        wire=kernel.observation(env,obs,prompt)
        stamp={k:GL.glGetString(getattr(GL,k)).decode() for k in ['GL_RENDERER','GL_VENDOR','GL_VERSION']}
        if not any(x in stamp['GL_RENDERER'].lower() for x in ('llvmpipe','softpipe','swrast')):
            raise RuntimeError('CPU renderer not attested')
        rec=Recorder(a.output,provenance=dict(kind='tiny_real_MT50_CPU_capture',synthetic=False,
            recorded=True,renderer=stamp,versions={k:importlib.metadata.version(k) for k in ['metaworld','mujoco','numpy']},
            task=task,init=0,env_seed=7,history_coverage='three zero-action-spaced frames, not a policy rollout'),limit=3)
        for i in range(3):
            if i:
                obs,*_=env.step(np.zeros(4,dtype=np.float32))
                wire=kernel.observation(env,obs,prompt)
            rec.add(wire,dict(campaign='xgpu_MT50_capture',benchmark_protocol=benchmark['protocol_identity'],
                task_name=task,task_id=0,original_init_idx=0,decision_seq=0,seed=7,
                explicit_seed=decision_seed(benchmark,task,0,i,7)))
        print(rec.close()['identity'])
    finally:env.close()


if __name__=='__main__':main()
