"""A100 worker-island probe: import the frozen LIBERO worker module, build one
real libero_spatial env (EGL), reset to an official init and render once."""
import json, os, sys, time
runtime = sys.argv[1]
t0 = time.time()
sys.path[:0] = [runtime + "/packages/openpi-client/src", runtime + "/src", runtime]
import exp.r17_integration.conductor.worker_entry, exp.r17_integration.benchmarks.libero_bridge, exp.libero_groot.emit_task_map
import examples.libero.main as M
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
env, desc = M._get_libero_env(task, 256, 0)
init = suite.get_task_init_states(0)
env.reset(); obs = env.set_init_state(init[0])
img = obs['agentview_image']
print(json.dumps(dict(ok=True, desc=desc, image=list(img.shape), mean=float(img.mean()), seconds=time.time() - t0)))
env.close()
