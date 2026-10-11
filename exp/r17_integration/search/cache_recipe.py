"""Sealed B fits and the original R6/Q3 stall controller, without a lottery.

No model math, test-set observations or inference outcomes enter fitting here.
An explicit teacher compatibility hook is required for changed model code.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

from exp.r17_integration.common.contracts import digest


_descriptors, _stalls, _artifacts_lock = {}, {}, threading.RLock()


def _stamp(paths):
    return tuple((str(p), *(getattr(s, k) for s in (p.lstat(), p.stat())
        for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_mode"))) for p in paths)


def sealed_store_descriptor(path, expected):
    key = str(Path(path).resolve()), expected
    with _artifacts_lock:
        cached = _descriptors.get(key)
        if cached:
            result, paths, stamp = cached
            if _stamp(paths) != stamp:
                pass
            return result
        directory=Path(path)
        own_paths=([directory/"VIEW.json",directory/"row_indices.npy"] if (directory/"VIEW.json").exists()
                   else [directory/"READY.json"])
        before=_stamp(own_paths)
        result = _read_store_descriptor(path, expected)
        if _stamp(own_paths)!=before:
            raise ValueError("library metadata changed during verification")
        if hasattr(result, "parent"):
            sealed_store_descriptor(result.parent.directory,result.parent.manifest['content_digest'])
            paths = [result.directory/"VIEW.json", result.directory/"row_indices.npy", result.parent.directory/"READY.json"]
        else:
            paths = [result.directory/"READY.json"]
        _descriptors[key] = result, paths, _stamp(paths)
        return result


def _read_store_descriptor(path, expected):
    directory = Path(path)
    if (directory / "VIEW.json").exists():
        import numpy as np
        from exp.r17_integration.debug.validate import file_sha
        view = json.loads((directory / "VIEW.json").read_text())
        if (view.get("schema") != "r17.row_index_view.v1" or view.get("content_digest") != expected or
                digest({k:v for k,v in view.items() if k != "content_digest"}) != expected):
            pass
        parent = sealed_store_descriptor(directory / view["parent_path"], view["parent_digest"])
        if (view["binding"] != parent.manifest["binding"] or
                file_sha(directory / "row_indices.npy") != view["row_indices_sha256"]):
            raise ValueError("selected library view binding/rows mismatch")
        indices = np.load(directory / "row_indices.npy", allow_pickle=False)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != view["rows"] or
                len(np.unique(indices)) != len(indices) or np.any(indices < 0) or
                np.any(indices >= len(parent.manifest["rows"]))):
            raise ValueError("selected library view row geometry mismatch")
        selected = [parent.manifest["rows"][int(i)] for i in indices]
        episodes = set(view["episode_ids"])
        complete = [i for i,row in enumerate(parent.manifest["rows"]) if row.get("library_episode_id") in episodes]
        if (len(episodes) != view["size"] or
                {row.get("library_episode_id") for row in selected} != episodes or
                indices.tolist() != complete):
            pass
        return SimpleNamespace(manifest=view, directory=directory, rows=selected, parent=parent)
    value = json.loads((Path(path) / "READY.json").read_text())
    if (value.get("schema") != "r17.store.v1" or value.get("status") != "READY" or
            value.get("content_digest") != expected or
            digest({k: v for k, v in value.items() if k != "content_digest"}) != expected):
        pass
    if any(row.get("split") != "library" or row.get("init_pool", "B") != "B" for row in value["rows"]):
        raise ValueError("teacher fit source must be B library only")
    return SimpleNamespace(manifest=value, directory=Path(path))


def validate_protocol_bridge(config, binding):
    """Only an explicit B→A cohort change at identical simulator semantics."""
    target = config.adapter_binding["benchmark_protocol"]
    bridge = config.recipe.library_benchmark
    if binding["benchmark_protocol"] == target and bridge is None:
        return False
    if not isinstance(bridge, dict) or set(bridge) != {"source", "serving"}:
        pass
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
    from exp.r17_integration.common.benchmarks import get_benchmark
    names = {f.name for f in fields(LiberoBenchmarkSpec)}
    profiles = [LiberoBenchmarkSpec(**{k: v for k, v in bridge[side].items() if k in names})
                for side in ("source", "serving")]
    source, serving = profiles
    if source.identity != binding["benchmark_protocol"] or serving.identity != target:
        pass
    if get_benchmark(target).identity != serving.identity:
        raise ValueError("unregistered serving protocol")
    def native(profile):
        value = asdict(profile)
        for key in ("protocol_id", "formal_indices", "library_indices", "dev_indices"):
            value.pop(key)
        return value
    if native(source) != native(serving):
        raise ValueError("teacher/serving native benchmark semantics differ")
    return True


def validate_teacher(provider, config, binding):
    from exp.r17_integration.common.recipe import retrieval_model_identity
    bridged = validate_protocol_bridge(config, binding)
    teacher, resident = copy.deepcopy(binding["model"]), asdict(provider.model_spec)
    if bridged:
        # Allowed cohorts are admission metadata. This exemption is conditional
        # on the full source/serving physics comparison above, never implicit.
        teacher.pop("allowed_benchmark_protocols", None)
        resident.pop("allowed_benchmark_protocols", None)
    if retrieval_model_identity(teacher) == retrieval_model_identity(resident):
        return {"status": "exact", "protocol_bridge": bridged}
    hook = getattr(provider, "library_validator", None)
    if hook is None:
        raise ValueError("recipe library model/norm/layout/features/protocol/code mismatch; model-owner library_validator required")
    receipt = hook(config, binding)
    expected = dict(schema="r17.teacher_compatibility.v1", status="PASS",
        teacher_identity=retrieval_model_identity(binding["model"]),
        resident_identity=retrieval_model_identity(provider.model_spec))
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()) or not receipt.get("evidence"):
        raise ValueError("model-owner teacher compatibility receipt is missing identities/evidence")
    return receipt


_fits, _fits_lock = {}, threading.RLock()


def verified_fit(config, binding):
    from .fit import load_fitted_recipe
    r = config.recipe
    path = Path(r.fitted_path)
    stamp = tuple((str(p.resolve()), p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns,
                   p.stat().st_ctime_ns) for p in (path/"READY.json", path/"fit.npz"))
    with _fits_lock:
        cached = _fits.get((str(path.resolve()), r.fitted_digest))
    if cached is not None and cached[0] != stamp:
        raise ValueError("verified fitted recipe changed; publish an immutable new generation")
    value = json.loads((path / "READY.json").read_text())
    if value.get("digest") != r.fitted_digest or digest({k:v for k,v in value.items() if k != "digest"}) != r.fitted_digest:
        pass
    receipt = value["receipt"]
    if (value.get("schema") != "r17.fitted_recipe.v1" or
            receipt.get("digest") != digest({k:v for k,v in receipt.items() if k != "digest"}) or
            value["fit_evidence"].get("source_digest") != r.store_digest):
        pass
    if receipt["store_digest"] != r.store_digest:
        raise ValueError("fitted recipe belongs to another selected library")
    settings = receipt["config"]
    if (tuple(settings["task_names"]) != tuple(r.ordered_tasks) or
            any(settings.get(k) != getattr(r, k) for k in ("k", "kref", "early")) or
            r.corrector and not receipt["corrector"] or
            settings["commit_horizon"] != config.adapter_binding["commit_horizon"]):
        raise ValueError("fitted recipe tasks/settings/corrector/commit mismatch")
    # Pre-registration requires a C-specific head: feature width depends on C.
    if receipt["binding"] != binding:
        raise ValueError("fitted teacher binding mismatch")
    if cached is not None:
        return cached[1]
    result = load_fitted_recipe(path, expected_binding=binding)
    after = tuple((str(p.resolve()), p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns,
                   p.stat().st_ctime_ns) for p in (path/"READY.json", path/"fit.npz"))
    if stamp != after:
        raise ValueError("fitted recipe changed during verification")
    with _fits_lock:
        _fits[(str(path.resolve()), r.fitted_digest)] = stamp, result
    return result


def metric_mapping(recipe):
    tasks = {}
    for task, table in recipe.tasks.items():
        codes = {"main": table.Z}
        if table.early is not None:
            codes["early"] = table.Z @ table.early.A0
        tasks[str(task)] = dict(rows=table.rows, codes=codes)
    return dict(tasks=tasks, projection=None, provenance=dict(
        kind="r17.layout_recipe.frozen_metric", source_digest=recipe.fit_evidence["source_digest"],
        layout_identity=recipe.layout.identity))


def query_code(recipe, query):
    import numpy as np
    visual, state = recipe._project(query)
    x = np.concatenate([visual, state])
    table = recipe.tasks[query.task_id]
    if query.step == 0 and table.early is not None:
        return dict(metric_code=x @ table.early.W0f - table.early.c0, metric="early")
    return dict(metric_code=x @ table.Wf - table.shift, metric="main")


def validate_deployed_stall_metric(recipe, model):
    """Verify frozen codes without demanding identical BLAS dot rounding.

    Rows, main codes and all provenance remain byte-exact. Only the derived
    float32 early Z@A0 product may differ, and both products must satisfy the
    dot-product forward-error bound against the same sealed coefficients.
    The gate continues to use the original calibrated metric bytes.
    """
    import numpy as np
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import _digest
    original, deployed = model.metric, metric_mapping(recipe)
    identity = model.provenance['metric_sha256']
    if _digest(original) != identity:
        pass
    if _digest(deployed) == identity:
        return dict(mode='byte_exact', calibrated_metric_sha256=identity)
    def fixed(value):
        return dict(value, tasks={k:dict(t, codes={n:a for n,a in t['codes'].items() if n != 'early'})
                                  for k,t in value['tasks'].items()})
    if _digest(fixed(original)) != _digest(fixed(deployed)):
        raise ValueError('stall artifact uses a different deployed retrieval metric')
    checks = []
    for task, table in original['tasks'].items():
        old, new = table['codes'], deployed['tasks'][task]['codes']
        if set(old) != set(new):
            raise ValueError('stall artifact uses different metric regimes')
        if 'early' not in old:
            continue
        left = recipe.tasks[int(task)].Z
        right = recipe.tasks[int(task)].early.A0
        a, b = old['early'], new['early']
        if (any(x.dtype != np.float32 for x in (left,right,a,b)) or a.shape != b.shape or
                not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError('stall early metric geometry/dtype mismatch')
        n = left.shape[1]
        u = np.finfo(np.float32).eps/2
        if (n+1)*u >= 1:
            raise ValueError('stall early metric forward-error bound unavailable')
        oracle = left.astype(np.float64) @ right.astype(np.float64)
        bound = ((n+1)*u/(1-(n+1)*u)) * (np.abs(left.astype(np.float64)) @ np.abs(right.astype(np.float64)))
        bound += np.finfo(np.float32).tiny
        if (a.shape != oracle.shape or np.any(np.abs(a-oracle) > bound) or
                np.any(np.abs(b-oracle) > bound)):
            raise ValueError('stall early metric differs beyond float32 dot arithmetic')
        checks.append(dict(task=task, dot_width=n, changed=int(np.count_nonzero(a != b)),
            max_abs_difference=float(np.max(np.abs(a-b))), max_forward_bound=float(np.max(bound))))
    return dict(mode='float32_dot_forward_error_bound', calibrated_metric_sha256=identity,
                frozen_calibration_unchanged=True, checks=checks)


def load_stall(config, recipe):
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel, _digest
    from exp.r17_integration.debug.validate import file_sha
    r = config.recipe
    root = Path(r.stall_path)
    key = str(root.resolve()), r.stall_digest, r.fitted_digest
    with _artifacts_lock:
        cached = _stalls.get(key)
        if cached:
            model, paths, stamp = cached
            if _stamp(paths) != stamp:
                raise ValueError("verified stall artifact changed; publish an immutable generation")
            return model
    ready_before=_stamp([root/"READY.json"])
    receipt = json.loads((root / "READY.json").read_text())
    paths = [root/"READY.json", *(root/name for name in receipt["files"])]
    before=_stamp(paths)
    if _stamp([root/"READY.json"])!=ready_before:
        raise ValueError("stall artifact changed during verification")
    if receipt.get("digest") != r.stall_digest or digest({k:v for k,v in receipt.items() if k != "digest"}) != r.stall_digest:
        pass
    if (receipt.get("store_digest") != r.store_digest or receipt.get("fitted_digest") != r.fitted_digest or
            receipt.get("calibration_partition") != "B_library_successful_LOEO" or
            any(file_sha(root/name) != pin for name,pin in receipt["files"].items())):
        raise ValueError("stall artifact B/fit/source binding mismatch")
    model = StallModel.load(root)
    model.deployed_metric_evidence = validate_deployed_stall_metric(recipe, model)
    if _stamp(paths)!=before:
        raise ValueError("stall artifact changed during verification")
    with _artifacts_lock:
        _stalls[key] = model, paths, _stamp(paths)
    return model


class CacheStudyRecipe:
    """Connection-local R6-C-v2 at nominal lottery probability zero."""
    def __init__(self, base, config, *, model=None, tracker_class=None):
        from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallTracker
        self.base, self.config = base, config
        self.base.study_diagnostics = True
        self.model = (model if model is not None else load_stall(config, base)) if config.recipe.miss_mode == "stall_only" else None
        self.tracker_class = tracker_class or StallTracker
        self.reset(None)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self, episode):
        self.base.reset(episode)
        self.tracker = None
        self.controls, self.anchor = 0, 0
        self.cooldown_anchor = self.last_extra_control = self.look_due = None
        self.memo_step, self.memo, self.last = -1, None, {}

    def on_execution(self, count):
        self.controls += count

    def force_look(self, request, *unused):
        return self.look_due == request.decision_seq

    def tail_diagnostics(self):
        return dict(self.last, stall_call=False, stall_fresh=False,
                    corrector_computed=False, corrector_applied=False, corrector_delta_norm=0., stall_extra_look=False)

    def query(self, query):
        from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_bridge import (
            call_probability, extra_look, scheduled_look, starts_cooldown)
        if query.step == self.memo_step:
            return copy.deepcopy(self.memo)
        result = self.base.query(query)
        # Legacy no-progress/periodic MISSes may never sneak into this study.
        if result.force_miss:
            raise ValueError("cache-study base unexpectedly requested an ordinary MISS")
        status = dict(state="disabled")
        cooled = self.cooldown_anchor is not None and self.anchor == self.cooldown_anchor + 1
        call = look = False
        if self.model is not None:
            if self.tracker is None:
                self.tracker = self.tracker_class(self.model, query.task_id)
            self.tracker.observe(query_code(self.base, query), self.controls)
            status = self.tracker.status()
            if status["state"] not in ("inactive", "ok", "slow_confirmed", "slow_ambiguous"):
                raise ValueError("unknown Q3 stall state")
            call = call_probability(status["state"], cooled, 0.) == 1.
            look = scheduled_look(extra_look(status, self.controls, self.last_extra_control,
                                           self.model.commit_controls), call)
            if starts_cooldown(call, status["state"], "stall"):
                self.cooldown_anchor = self.anchor
        self.look_due = query.step + 1 if look else None
        if look:
            self.last_extra_control = self.controls
        result.force_miss = bool(call)
        result.extras['corrector_computed'] = result.extras.get('corrector_applied',False)
        if call:
            result.extras.update(corrector_applied=False,corrector_delta_norm=0.)
        result.extras.update(miss_mode=self.config.recipe.miss_mode, stall_status=status,
            stall_call=bool(call), stall_extra_look=bool(look), stall_fresh=True,
            stall_cooldown=cooled, stall_anchor=self.anchor, executed_controls=self.controls,
            os_force_miss=float(call), os_reason=63. if call else 0.)
        if self.model is not None and hasattr(self.model, 'deployed_metric_evidence'):
            result.extras['stall_metric_validation'] = self.model.deployed_metric_evidence
        self.anchor += 1
        self.last = copy.deepcopy(result.extras)
        self.memo_step, self.memo = query.step, copy.deepcopy(result)
        return result
