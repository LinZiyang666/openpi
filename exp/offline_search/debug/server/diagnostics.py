"""Read-only adapters and copies of already computed metric intermediates."""
import contextlib
import dataclasses
import functools
import math
import types

import numpy as np

from ..schema import status


def method_chain(method):
    seen = set()
    while method is not None and id(method) not in seen:
        seen.add(id(method))
        yield method
        method = getattr(method, "base", None)


def detached(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def split_diagnostics(value, arrays, prefix="diag", path=()):
    """JSON structure retains references to full numeric diagnostic arrays."""
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        key = prefix + "_" + "_".join(path)
        array = detached(value)
        if array.dtype.hasobject:
            raise TypeError("object diagnostic array at " + key)
        # Collision-free references preserve arbitrary method field names.
        original = key
        suffix = 1
        while key in arrays:
            key = original + "_" + str(suffix)
            suffix += 1
        arrays[key] = array
        return {"array": key}
    if dataclasses.is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): split_diagnostics(v, arrays, prefix, path + (str(k),)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            array = np.asarray(value)
            if array.dtype.kind in "biufc":
                return split_diagnostics(array, arrays, prefix, path)
        return [split_diagnostics(v, arrays, prefix, path + (str(i),)) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError("unsupported diagnostic type: " + type(value).__name__)


def debug_record(method):
    """Protocol first, then adapters for A, CU/CT, SF/SW and SeededInference.

    Adapter reads never call tracker.status, plan/check, retrieval or RNG. New
    debug_record methods are contractually read-only and called after response.
    """
    if method is None:
        return {}, status("not_applicable", "no serving method")
    protocol = getattr(method, "debug_record", None)
    if callable(protocol):
        value = protocol()
        if not isinstance(value, dict):
            raise TypeError("debug_record must return a dict")
        return value, status("available", "method debug_record protocol")
    modules = (".blind_awm", ".g1_awm.awm", ".method_c.methods", ".c3_calls.methods",
               ".c1_follow.methods", ".c2_wrist.method", ".k4_eval.seeded_inference")
    chain = list(method_chain(method))
    supported = any(any(type(x).__module__.endswith(name) for name in modules) for x in chain)
    out = {"method_class": type(method).__module__ + ":" + type(method).__name__}
    for i, obj in enumerate(chain):
        fields = {}
        for name in ("_last_log", "_tilt_log", "_plan_extras", "last_blind_extras", "_follow_plan",
                     "_deviation_latched", "_look_due_step", "_anchor_index", "_last_cooldown_anchor",
                     "_last_extra_control", "parameter", "lambda_", "budget", "random_seed", "randomization_key",
                     "next_camera_mode", "_camera_mode", "_policy_gate_anchor"):
            if hasattr(obj, name):
                fields[name] = getattr(obj, name)
        anchor = getattr(obj, "_anchor", None)
        if isinstance(anchor, dict):
            fields["anchor"] = {k: anchor[k] for k in ("rows", "weights", "phase", "rs", "task", "episode", "step",
                                                    "last_step") if k in anchor}
        if fields:
            out["layer_" + str(i)] = fields
    return out, status("available" if supported else "unsupported",
                       "read-only existing-method adapter" if supported else "no registered method adapter")


@contextlib.contextmanager
def metric_tap(session, target):
    """Temporarily wrap only this connection's live metric/mixture methods.

    AWM._dist exposes xv/rs8 in its existing return tuple. _mix supplies exact
    weights even when a MISS invalidates the anchor. Bound wrappers retain
    __func__ so C's shallow metric facade invokes the original arithmetic on
    that facade. SW's temporary field substitution is also observed directly.
    No profiler is installed, and no callback runs during model forwards.
    """
    restored = []
    missing = object()
    def wrap(fn, name):
        unbound = getattr(fn, "__func__", None)
        @functools.wraps(fn)
        def captured(owner, *args, **kwargs):
            result = unbound(owner, *args, **kwargs) if unbound is not None else fn(*args, **kwargs)
            try:
                if name == "_dist" and isinstance(result, tuple) and len(result) == 11:
                    xv, rs = result[5], result[6]
                    width0 = getattr(getattr(owner, "B0T", None), "shape", (0,))[0]
                    target["keys_pca_third"] = detached(xv[:width0], np.float32) if width0 else None
                    target["keys_pca_wrist"] = detached(xv[width0:], np.float32)
                    target["metric_state"] = detached(rs, np.float32)
                elif name == "_mix" and isinstance(result, tuple) and len(result) == 3:
                    rows = args[0] if args else kwargs["rows"]
                    target["weights"] = detached(result[0], np.float32)
                    target["rows"] = detached(rows, np.int64)
            except Exception as exc:
                target["tap_error"] = "{}: {}".format(type(exc).__name__, exc)
            return result
        return captured
    try:
        for obj in method_chain(getattr(session, "method", None)):
            for name in ("_dist", "_mix"):
                fn = getattr(obj, name, None)
                if not callable(fn):
                    continue
                previous = vars(obj).get(name, missing)
                setattr(obj, name, types.MethodType(wrap(fn, name), obj))
                restored.append((obj, name, previous))
        yield
    finally:
        for obj, name, previous in reversed(restored):
            if previous is missing:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
