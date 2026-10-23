"""Isolated per-stage GPU time of the cluster-simulation π0.5 server at each batch size (one stage at a time).

Same model, patches and CUDA graphs as serve_entry.py; the observation is the first infer frame of one recorded
episode, repeated to the batch size. Run with the server tree's python and PYTHONPATH (see serve.py):

  CLUSTER_SIM_GRAPHS=1 python exp/cluster_sim/bench_stages.py --library /nvme/weiland/openpi/cluster_sim/library
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CLUSTER_SIM_LOG', '/tmp/cluster_sim_bench')
sys.path.insert(0, str(Path(__file__).parent))
import serve_entry as se  # noqa: E402  (installs the same patches as the server)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--sizes', default='1,2,4,8,12,16,24,32')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    import msgpack
    from openpi_client import msgpack_numpy
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.training import config as _config

    doc = msgpack.unpackb(next(Path(args.library).glob('ep_*.msgpack')).read_bytes(), raw=False)
    frame = next(f for f, k in zip(doc['frames'], doc['kinds']) if k == 'infer')
    obs = {k: v for k, v in msgpack_numpy.unpackb(frame).items() if not k.startswith('__')}
    policy = policy_config.create_trained_policy(_config.get_config('pi05_libero'), se_ckpt())
    model, device = policy._model, policy._pytorch_device
    inputs = policy._input_transform(dict(obs))
    inputs = {k: v for k, v in inputs.items()}
    import jax
    one = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    se._gate.enter()
    rows = {}
    for b in map(int, args.sizes.split(',')):
        batch = _model.Observation.from_dict(jax.tree.map(lambda t: t.expand(b, *t.shape[1:]).contiguous(), one))
        noise = torch.randn(b, model.config.action_horizon, model.config.action_dim, device=device)
        times = {}
        with torch.no_grad():
            for name, fn in (('stage1', lambda: model.run_stage1(batch)),):
                s1 = fn()
            s2 = model.run_stage2(s1)
            for _ in range(2):
                model.run_stage3(s2, noise=noise, num_steps=10, return_intermediates=True)
            cases = {'stage1': lambda: model.run_stage1(batch), 'stage2': lambda: model.run_stage2(s1),
                     'stage3_k10': lambda: model.run_stage3(s2, noise=noise, num_steps=10, return_intermediates=True),
                     'stage3_k1': lambda: model.run_stage3(s2, noise=noise, num_steps=1, return_intermediates=True)}
            for name, fn in cases.items():
                fn()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(args.iters):
                    fn()
                end.record()
                end.synchronize()
                times[name] = round(start.elapsed_time(end) / args.iters, 2)
        times['pure_per_request'] = round((times['stage1'] + times['stage2'] + times['stage3_k10']) / b, 2)
        rows[b] = times
        print(b, json.dumps(times), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(graphs=se.GRAPHS, compile=se.COMPILE, text_tokens=se.TEXT_TOKENS,
                                                  rows=rows), indent=1) + '\n')


def se_ckpt():
    return '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'


if __name__ == '__main__':
    main()
