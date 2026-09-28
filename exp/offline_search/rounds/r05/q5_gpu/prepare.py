"""Rebase the small opt-in patch onto the installed Q2 plugin, without installing."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
REPO=next(p for p in B.parents if (p/'exp/trace_dual/config').is_dir())
source=REPO/'exp/offline_search/closed_loop/plugin.py'
s=source.read_text()
(B/'before/plugin.py').write_text(s)
(B/'results/preimage.json').write_text(json.dumps({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
def change(old,new):
    global s
    assert s.count(old)==1,(old[:80],s.count(old))
    s=s.replace(old,new)
change('    return ap\n', '''    ap.add_argument("--os-gpu-retrieval", choices=("shadow", "serve"), default=None,
                    help="opt-in K9 CUDA graph retrieval; shadow keeps CPU actions, serve replaces retrieval")
    return ap
''')
change('        self.randomized = getattr(opts, "os_rand_seed", None) is not None', '''        self.gpu = None
        if getattr(opts, "os_gpu_retrieval", None):
            from exp.offline_search.rounds.r05.q5_gpu.dev.gpu_retrieval import validate_options
            validate_options(opts, model)
        self.randomized = getattr(opts, "os_rand_seed", None) is not None''')
change('        if self.randomized:\n            from exp.offline_search.rounds.r04.k5_rand.overlay import validate_method', '''        if getattr(opts, "os_gpu_retrieval", None):
            from exp.offline_search.rounds.r05.q5_gpu.dev.gpu_retrieval import DeviceRuntime
            self.gpu = DeviceRuntime(self)
        if self.randomized:
            from exp.offline_search.rounds.r04.k5_rand.overlay import validate_method''')
change('        self.emit(row)\n        log.info("osplug ready:', '''        if self.gpu is not None:
            row["gpu_retrieval"] = self.gpu.info
        self.emit(row)
        log.info("osplug ready:''')
change('        if self.native_mode:\n            strategies[cp] = NativeProxy(native, sess)', '''        if self.gpu is not None:
            from exp.offline_search.rounds.r05.q5_gpu.dev.gpu_retrieval import attach
            attach(self, sess, comps)
        if self.native_mode:
            strategies[cp] = NativeProxy(native, sess)''')
change('        pp = getattr(getattr(config.backend, "in_memory", None), "preload_path", None)', '''        if self.gpu is not None:
            if any(k != "cp1" and c.enabled for k, c in config.checkpoints.items()):
                raise RuntimeError("GPU retrieval supports only CP1")
            if getattr(config.write_policy, "type", None) != "never":
                raise RuntimeError("GPU retrieval requires write_policy=never")
        pp = getattr(getattr(config.backend, "in_memory", None), "preload_path", None)''')
change('        self.lock = threading.RLock() if rt.r4 else contextlib.nullcontext()',
       '        self.lock = threading.RLock() if rt.r4 or rt.gpu is not None else contextlib.nullcontext()')
change('clone_method(rt.method, strict=rt.r4)', 'clone_method(rt.method, strict=rt.r4 or rt.gpu is not None)')
change('        self.b_v0.append(_np32(qk["vision_0"]).reshape(-1))\n        self.b_v1.append(_np32(qk["vision_1"]).reshape(-1))', '''        if self.rt.gpu is not None and self.rt.opts.os_gpu_retrieval == "serve":
            self.b_v0.append(np.full(self.rt.dims.KEY_DIM, np.nan, np.float32))
            self.b_v1.append(np.full(self.rt.dims.KEY_DIM, np.nan, np.float32))
        else:
            self.b_v0.append(_np32(qk["vision_0"]).reshape(-1))
            self.b_v1.append(_np32(qk["vision_1"]).reshape(-1))''')
change('        self._push_inputs(ctx)\n        step = self.step', '''        self._push_inputs(ctx)
        if rt.gpu is not None:
            self._gpu_host_keys_ms = (time.perf_counter_ns() - t_all) / 1e6
        step = self.step''')
change('        t_all = time.perf_counter_ns()\n        self._push_inputs(ctx)', '''        t_all = time.perf_counter_ns()
        if rt.gpu is not None and ctx.current_step == 0 and self.step > 0:
            raise RuntimeError("GPU retrieval needs an explicit episode reset before implicit restart")
        self._push_inputs(ctx)''')
change('        res = self.method.query(view)', '''        gpu_diag = None
        if rt.gpu is None:
            res = self.method.query(view)
        else:
            from exp.offline_search.rounds.r05.q5_gpu.dev.gpu_retrieval import query
            res, gpu_diag = query(self, view)''')
change('        if rt.r4:\n            anchor_owner = self.method', '''        if gpu_diag is not None:
            self._dec["gpu_retrieval"] = gpu_diag
        if rt.r4:
            anchor_owner = self.method''')
change('        if err:\n            row["error"]', '''        if "gpu_retrieval" in d:
            row["gpu_retrieval"] = d["gpu_retrieval"]
        if err:
            row["error"]''')
change('        enabled = [s for s in sessions if s.rt.r4]', '''        gpu_sessions = [s for s in sessions if s.rt.gpu is not None]
        if gpu_sessions:
            if len(sessions) != 1:
                raise RuntimeError("GPU retrieval expects one CP1 session per connection")
            self._osp_lock = gpu_sessions[0].lock
        enabled = [s for s in sessions if s.rt.r4]''')
change('            for session in sessions:\n                if session.rt.policy_tail', '''            for session in sessions:
                if session.rt.gpu is not None:
                    session._gpu_failed = True
                if session.rt.policy_tail''')
(B/'dev/plugin.py').write_text(s)
