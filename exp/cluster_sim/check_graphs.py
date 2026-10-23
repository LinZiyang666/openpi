"""Check and time the whole-stage graphs of serve_entry.py against the eager model (same observation, same noise).

Builds coordinator payloads from the first infer frame of a recorded episode, runs Pi05StageBatcher.run_stage1_batch
and the fused run_stage3_miss (graphs) for each batch size, then the eager model (instance patches removed) on the
same inputs; prints the max |action| difference and the graphed per-batch times. Run with the server tree's python:

  CLUSTER_SIM_GRAPHS=1 CLUSTER_SIM_FUSE23=1 CLUSTER_SIM_COMPILE=all CLUSTER_SIM_COMPILE_MODE=default \
      python exp/cluster_sim/check_graphs.py --library /nvme/weiland/openpi/cluster_sim/library --steps 10
"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CLUSTER_SIM_LOG', '/tmp/cluster_sim_check')
sys.path.insert(0, str(Path(__file__).parent))
import serve_entry as se  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--sizes', default='1,4,16')
    args = parser.parse_args()
    import jax
    import msgpack
    from openpi_client import msgpack_numpy
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.serving import batching_coordinator as bco
    from openpi.serving.batching_core import Stage3MissPayload
    from openpi.training import config as _config

    doc = msgpack.unpackb(next(Path(args.library).glob('ep_*.msgpack')).read_bytes(), raw=False)
    frame = next(f for f, k in zip(doc['frames'], doc['kinds']) if k == 'infer')
    obs = {k: v for k, v in msgpack_numpy.unpackb(frame).items() if not k.startswith('__')}
    policy = policy_config.create_trained_policy(_config.get_config('pi05_libero'),
                                                 '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch')
    model, device = policy._model, policy._pytorch_device
    payload = jax.tree.map(lambda x: torch.from_numpy(np.array(x)), policy._input_transform(dict(obs)))
    batcher = bco.Pi05StageBatcher(model, device)
    se._gate.enter()
    results = {}
    with torch.no_grad():
        for b in map(int, args.sizes.split(',')):
            noise = torch.randn(b, model.config.action_horizon, model.config.action_dim, device=device)
            def graphed():
                shards = batcher.run_stage1_batch([payload] * b)
                pending = []
                for i, shard in enumerate(shards):
                    s2 = se._pi0.Stage2Output(stage1=shard, past_key_values=None)
                    s2.cs_pending = True
                    pending.append(Stage3MissPayload(stage2_out=s2, noise=noise[i], num_steps=args.steps))
                return batcher.run_stage3_miss(pending, num_steps=args.steps, save_timesteps_per_request=[None] * b)
            replies = graphed()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                graphed()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 100
            results[b] = (torch.cat([r.action_chunk for r in replies]), ms)
    # Eager reference: drop the graphed instance attributes, run the model's own staged methods.
    for name in ('_stage2_llm_backbone', 'denoise_step', 'run_stage2_capture'):
        model.__dict__.pop(name, None)
    print('graphed per-batch ms (stage 1 + fused back end, wall, incl. CPU):', {b: round(ms, 1) for b, (_, ms) in results.items()})
    with torch.no_grad():
        b = 1
        torch.manual_seed(1)
        noise = torch.randn(b, model.config.action_horizon, model.config.action_dim, device=device)
        shards = batcher.run_stage1_batch([payload])
        s2 = se._pi0.Stage2Output(stage1=shards[0], past_key_values=None)
        s2.cs_pending = True
        g = batcher.run_stage3_miss([Stage3MissPayload(stage2_out=s2, noise=noise[0], num_steps=args.steps)],
                                    num_steps=args.steps, save_timesteps_per_request=[None])[0].action_chunk
        tree = jax.tree.map(lambda t: t[None].to(device), payload)
        o = _model.Observation.from_dict({**tree, 'image': dict(tree['image'])})
        s1 = model.__class__.run_stage1(model, o)
        e = model.__class__.run_stage3(model, model.__class__.run_stage2(model, s1), noise=noise, num_steps=args.steps,
                                       return_intermediates=True).action_chunk
        print('max |graphed - eager| action:', float((g - e).abs().max()), 'action scale', float(e.abs().max()))


if __name__ == '__main__':
    main()
