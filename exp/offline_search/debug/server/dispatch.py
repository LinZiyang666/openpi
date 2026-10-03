"""Count connection-owned live stage dispatches; never add a model call."""
import functools
import time

from ..schema import status


def install_dispatch_tap(inner, session):
    obj, seen = inner, set()
    while id(obj) not in seen:
        seen.add(id(obj))
        fields = vars(obj)
        if "_orchestrator" in fields or hasattr(obj, "_osp_prepare_blind"):
            break
        nxt = next((fields[k] for k in ("_osp_inner", "_inner", "_policy") if k in fields), None)
        if nxt is None:
            return status("unsupported", "no supported connection-owned stage adapter")
        obj = nxt
    fake = hasattr(obj, "_osp_prepare_blind")
    if fake:
        targets = [(obj, "stage1", "stage1")]
    elif session.rt.model == "pi05":
        targets = [(obj, "_stage1_fn", "stage1"), (obj, "_stage2_fn", "stage2"),
                   (obj, "_stage3_fn", "stage3"), (obj, "_stage3_from_fn", "stage3")]
    elif session.rt.model == "groot":
        runner = obj._runner
        targets = [(runner, "run_stage1", "stage1"), (runner, "run_stage2", "stage23"),
                   (runner, "run_stage2_llm", "stage2"), (runner, "run_stage3", "stage3"),
                   (runner, "run_stage3_from", "stage3")]
    else:
        return status("unsupported", "no model stage adapter")

    def wrap(fn, stage):
        @functools.wraps(fn)
        def counted(*args, **kwargs):
            capture = getattr(session, "_debug_capture", None)
            started = time.perf_counter()
            if capture is not None:
                capture.dispatch[stage] = capture.dispatch.get(stage, 0) + 1
            try:
                return fn(*args, **kwargs)
            finally:
                if capture is not None:
                    elapsed = (time.perf_counter() - started) * 1000
                    capture.stage_ms[stage] = capture.stage_ms.get(stage, 0.) + elapsed
        return counted

    wrapped = []
    for owner, key, stage in targets:
        fn = getattr(owner, key, None)
        if callable(fn):
            setattr(owner, key, wrap(fn, stage))
            wrapped.append(stage)
    return status("available" if not fake and "stage1" in wrapped else "unsupported",
                  "CPU fake has stage1 only" if fake else "per-request live dispatch; coordinator may batch kernels")
