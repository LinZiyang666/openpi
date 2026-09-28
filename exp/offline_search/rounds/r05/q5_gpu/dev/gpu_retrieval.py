"""Opt-in K9 serving bridge. Fitted buffers shared; streams/graphs/history per connection.

This module is imported only with --os-gpu-retrieval. No src patches or global
Torch precision changes. CPU guards consume the small GPU feature packet.
"""
from __future__ import annotations
import math
import threading
import time
from types import SimpleNamespace
import numpy as np

UNSUPPORTED_BLIND = ('K9 does not implement blind anchor updates or K7 endpoint-confirmed '
                     'stuck features across visual gaps; K7/BlindAWM/policy-tail are unsupported')


def validate_options(opts, model):
    if model != 'pi05' or opts.os_method == 'native':
        raise ValueError('--os-gpu-retrieval requires a pi05 AWM/MixedJudge method')
    if opts.os_blind or opts.os_policy_tail:
        raise ValueError(UNSUPPORTED_BLIND)
    if not opts.os_no_shadow_native:
        raise ValueError('--os-gpu-retrieval requires --os-no-shadow-native')
    if opts.os_tokens != 'off':
        raise ValueError('--os-gpu-retrieval requires --os-tokens off')
    if opts.os_log_inputs and opts.os_gpu_retrieval == 'serve':
        raise ValueError('GPU serve cannot log host visual inputs; use shadow for --os-log-inputs')
    if getattr(opts, 'os_stage1_mode', 'full') != 'full':
        raise ValueError('GPU retrieval supports full-camera stage 1 only')
    if getattr(opts, 'os_log_r4', False) or getattr(opts, 'os_rand_seed', None) is not None:
        raise ValueError('GPU retrieval does not support R4 stage synchronization or randomized overlays')
    if getattr(opts, 'judge', None) is not None and opts.judge.mode == 'quantile':
        raise ValueError('GPU retrieval currently supports per-connection verdicts only; quantile is unsupported')


def validate_method(method):
    # Exact implementation allowlist: accepting arbitrary subclasses silently drops hooks.
    import inspect
    from pathlib import Path
    root = next(p for p in Path(__file__).resolve().parents if (p/'exp/trace_dual/config').is_dir())
    b = getattr(method, 'base', method)
    if hasattr(method, 'blind_step') or hasattr(b, 'blind_step'):
        raise ValueError(UNSUPPORTED_BLIND)
    expected = root / 'exp/offline_search/rounds/r02/g1_awm/awm.py'
    if type(b).__name__ != 'AWM' or Path(inspect.getfile(type(b))).resolve() != expected:
        raise ValueError('GPU retrieval requires the stock joint AWM implementation')
    if method is not b:
        expected = root / 'exp/offline_search/rounds/r03/h3_judge/judge.py'
        if type(method).__name__ != 'MixedJudge' or Path(inspect.getfile(type(method))).resolve() != expected:
            raise ValueError('GPU retrieval supports only stock guard-only MixedJudge')
        if (method.blend or method.recover or method.events or method.burst or not method.guards
                or method.T != 1.0):
            raise ValueError('GPU MixedJudge needs guards, T=1, no blend/recovery/events/burst')
    if (b.model != 'pi05' or b.features != 'joint' or b.feat0 != 'joint' or b.k != 16
            or not b.early or b.codes or b.norm_cap or b.hyst
            or any(t.As0 is not None or t.Z0 is not None for t in b.tasks.values())):
        raise ValueError('K9 needs pi05 joint k=16/full-rank early AWM without insurance')


class DeviceRuntime:
    def __init__(self, rt):
        from exp.offline_search.rounds.r04.k9_gpu_retrieval.gpu_awm import GPUAWM
        validate_options(rt.opts, rt.model)
        validate_method(rt.method)
        b = getattr(rt.method, 'base', rt.method)
        lib = rt.store.LibraryView(rt.root, rt.lib_key, b.cand_name)
        try:
            # PI0Pytorch.__init__ calls set_float32_matmul_precision('high').
            # FP64 projection/distance GEMVs avoid TF32 without changing a
            # process-global setting while another connection runs the model.
            self.module = GPUAWM(rt.method, lib, precision='float64').eval()
        except AssertionError as exc:
            raise ValueError('fit exceeds the K9 supported shape/feature contract') from exc
        self.lock = threading.Lock()  # setup only; never held across an inference
        self.device = None
        self.info = dict(mode=rt.opts.os_gpu_retrieval, precision='float64',
                         resident_bytes=self.module.resident_bytes(), graph='per_connection_retrieval',
                         stage1_shared_graph=False,
                         stage1_graph_reason='served staged/coordinator stack uses max-autotune-no-cudagraphs; '
                         'joint capture requires a model/coordinator change',
                         chunk_atol=1e-4, confidence_atol=1e-3)
        self.info['event_boundary'] = 'input copies + retrieval graph; gpu_path_event_ms also includes pooling'
        self.info['shadow_host_keys'] = 'CPU reference still copies visual keys; serve never does'

    def connection(self):
        import torch
        with self.lock:
            if not torch.cuda.is_available():
                raise RuntimeError('--os-gpu-retrieval requires CUDA')
            device = torch.device('cuda', torch.cuda.current_device())
            if self.device is None:
                self.module.to(device)
                self.device = device
            if self.device != device:
                raise RuntimeError('GPU retrieval runtime is bound to one CUDA device')
            return GraphConnection(self.module, device)


class GraphConnection:
    def __init__(self, module, device):
        import torch
        self.module, self.device = module, device
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.done = torch.cuda.Event()
        shapes = [(1,32768),(1,32768),(1,32),(1,),(1,),(1,),
                  (1,10,32),(1,32768),(1,32768),(1,32),(1,)]
        self.inputs = [torch.zeros(sh, device=device, dtype=torch.int64 if i in (3,4,5,10) else torch.float32)
                       for i,sh in enumerate(shapes)]
        self.host_action_in = torch.empty((1,10,32), pin_memory=True)
        self.host_meta = torch.empty(3, dtype=torch.int64, pin_memory=True)
        # Inputs were zeroed asynchronously on the constructing thread's stream.
        # Publish those writes before warmup/capture on the connection stream.
        self.stream.wait_stream(torch.cuda.current_stream(device))
        def work():
            out = module(*self.inputs)
            names = [k for k,v in out.items() if v.ndim == 1]
            packet = torch.cat([out[k].double().reshape(-1) for k in names] +
                               [out['topk'].double().reshape(-1), out['scores'].reshape(-1).double(),
                                self.inputs[2].double().reshape(-1)])
            self.inputs[7].copy_(self.inputs[0]); self.inputs[8].copy_(self.inputs[1])
            self.inputs[9].copy_(self.inputs[2])
            if module.judge:
                self.inputs[10].copy_(out['stuck_n'])
            return out['action'], packet, names
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for _ in range(3):
                work()
            self.stream.synchronize()  # setup only, no device-wide barrier
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream, capture_error_mode='thread_local'):
                self.action, self.packet, self.names = work()
        self.host_action = torch.empty_like(self.action, device='cpu', pin_memory=True)
        self.host_packet = torch.empty_like(self.packet, device='cpu', pin_memory=True)
        self.copy_bytes = self.host_action.numel()*4 + self.host_packet.numel()*8
        self.last = None

    def run(self, keys, *, step, task, prev_hit, prev_action):
        import torch
        t0 = time.perf_counter_ns()
        for name, size in (('vision_0',32768),('vision_1',32768),('robot_state',32)):
            x = keys[name]
            if x.device != self.device or x.numel() != size:
                raise ValueError(f'GPU retrieval {name} must be resident on {self.device} with {size} elements')
        self.host_meta.copy_(torch.tensor([task,step,prev_hit], dtype=torch.int64))
        self.host_action_in.numpy()[0] = 0 if prev_action is None else prev_action
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.begin.record()
            for dst,name in zip(self.inputs[:3], ('vision_0','vision_1','robot_state')):
                dst.copy_(keys[name].reshape_as(dst), non_blocking=True)
            for i,dst in enumerate(self.inputs[3:6]):
                dst.copy_(self.host_meta[i:i+1], non_blocking=True)
            self.inputs[6].copy_(self.host_action_in, non_blocking=True)
            if step == 0:
                for dst in self.inputs[7:]:
                    dst.zero_()
            self.graph.replay()
            self.end.record()
            self.host_action.copy_(self.action, non_blocking=True)
            self.host_packet.copy_(self.packet, non_blocking=True)
            self.done.record()
        # The only inference host wait: both small final copies have completed.
        self.done.synchronize()
        p = self.host_packet.numpy(); n = len(self.names)
        out = {k: float(p[i]) for i,k in enumerate(self.names)}
        out.update(action=self.host_action.numpy()[0].copy(), topk=p[n:n+16].astype(np.int64),
                   scores=p[n+16:n+32].copy(), rs=p[n+32:].astype(np.float32))
        timing = dict(event_ms=self.begin.elapsed_time(self.end),
                      wall_ms=(time.perf_counter_ns()-t0)/1e6, final_d2h_bytes=self.copy_bytes)
        self.last = (out, timing)
        return self.last


class ResidentKeys:
    """Per-connection CP1 builder. Retrieval completes before orchestrator state D2H."""
    def __init__(self, inner, session, graph):
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        if type(inner) is not CP1SpatialPool16KeyBuilder:
            raise ValueError('GPU retrieval requires the stock CP1SpatialPool16KeyBuilder')
        self.inner, self.session, self.graph = inner, session, graph
        self.pool_event = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def build(self, checkpoint_id):
        import torch
        from openpi.cache.types import CheckpointID
        if checkpoint_id != CheckpointID.CP1:
            raise ValueError('GPU retrieval supports CP1 only')
        if self.session.__dict__.get('_gpu_failed') and not self.session.pending:
            raise RuntimeError('GPU decision failed; start a new episode before reusing this connection')
        raw = self.inner._slice()
        if raw['vision_0'].is_cuda:
            if self.pool_event is None:
                self.pool_event = torch.cuda.Event(enable_timing=True)
            self.pool_event.record()
        keys = {k:self.inner._reduce_vision(raw[k]) for k in ('vision_0','vision_1')}
        keys['robot_state'] = raw['robot_state']
        s = self.session
        if s.pending or s.ep is None:
            task = s.ident.get('task') or str((s.cur_obs or {}).get('prompt', ''))
            s._begin(SimpleNamespace(task_key=task, query_keys=keys))
            s._gpu_failed = False
        if s.b_aex.n != s.step:
            raise RuntimeError('GPU retrieval execution history is not committed')
        hit = s.hits[-1] if s.step else -1
        out, timing = self.graph.run(keys, step=s.step, task=s.ep.task_id, prev_hit=hit,
                                    prev_action=s.b_aex.a[s.step-1] if s.step else None)
        s._gpu_pending = (out, timing)
        if self.pool_event is not None:
            timing['gpu_path_event_ms'] = self.pool_event.elapsed_time(self.graph.end)
        # State is included in the final packet; orchestrator's .cpu() is now a no-op.
        keys['robot_state'] = torch.from_numpy(out['rs'])
        return keys


def attach(rt, session, comps):
    graph = rt.gpu.connection()
    kb = ResidentKeys(comps['key_builder'], session, graph)
    comps['key_builder'] = session.kb = kb


def cpu_guards(method, q, out):
    """Stock MixedJudge guard-only state machine fed exclusively by GPU features."""
    s = method._s
    step = int(q.step)
    method._memo_sync(s, step)
    s['prog'].append((out['top1_prog'], float(max(int(out['ep_len'])-1, 1))))
    noprog = 0
    for j in range(step, 0, -1):
        (a,n), (b,_) = s['prog'][j], s['prog'][j-1]
        if not (math.isfinite(a) and math.isfinite(b)) or (a-b)*n > method.prog_eps:
            break
        noprog += 1
    gexec = 0. if not step else (1. if q.prev_a_exec[4,6] >= 0 else -1.)
    flags = 0
    if out['stuck_n'] >= method.stuck_thr: flags |= 1
    if out['term1'] and gexec > 0: flags |= 2
    if out['overtime'] > 1 and out['lag'] > method.lag_thr and out['stuck_n'] >= 1: flags |= 4
    if noprog >= method.noprog_n-1: flags |= 8
    s['flag'].append(int(bool(flags)))
    s['stuck_n'] = int(out['stuck_n'])
    s['prev_top1'] = int(out['topk'][0])
    ex = {k:out[k] for k in ('pred_err','zsum','regime','stuck_n','overtime','lag','term1',
                              'vote','disp','dnn','vis','top1_prog')}
    ex.update(os_force_miss=float(bool(flags)), os_reason=float((flags & -flags).bit_length()),
              os_flags=float(flags), os_phase=0., os_conf_raw=out['confidence'], gexec=gexec,
              gprop=1. if out['action'][0,6] >= 0 else -1., noprog_n=float(noprog), burst_left=0.)
    if step: ex.update(motion=out['motion'], vself=out['vself'])
    return ex


def result(method, q, out):
    from exp.offline_search.harness import api
    b = getattr(method, 'base', method)
    if method is b:
        names = ['d1','d1_rel','disp5','dst','regime','lib_ep','lib_step','w_eff']
        if int(out['regime']) == 1: names.append('c0')
        if q.step: names.append('still')
        ex = {k:out[k] for k in names}
    else:
        ex = cpu_guards(method, q, out)
    return api.Result(out['topk'], out['scores'], out['confidence'], action=out['action'],
                      library=b.cand_name, extras=ex)


def query(session, view):
    out, timing = session._gpu_pending
    mode = session.rt.opts.os_gpu_retrieval
    t = time.perf_counter_ns()
    if mode == 'shadow':
        res = session.method.query(view)
        cpu_ms = (time.perf_counter_ns()-t)/1e6
        delta = float(np.max(np.abs(res.action-out['action'])))
        conf = abs(float(res.confidence)-out['confidence'])
        diag = dict(mode=mode, **timing, cpu_query_ms=cpu_ms,
                    gpu_top1=int(out['topk'][0]), gpu_top16=out['topk'].tolist(),
                    top1_agree=bool(res.topk[0]==out['topk'][0]),
                    top16_set_agree=np.array_equal(np.sort(res.topk),np.sort(out['topk'])),
                    top16_order_agree=np.array_equal(res.topk,out['topk']),
                    chunk_max_abs=delta, chunk_agree=bool(delta<=1e-4),
                    confidence_abs=conf, confidence_agree=bool(conf<=1e-3))
    else:
        res = result(session.method, view, out)
        diag = dict(mode=mode, **timing, cpu_query_ms=None,
                    cpu_guard_ms=(time.perf_counter_ns()-t)/1e6)
    diag['cpu_path_ms'] = None if mode == 'serve' else diag['cpu_query_ms'] + session._gpu_host_keys_ms
    return res, diag
