"""Frozen A retrieval and same-library third-camera fits for deferred inputs."""
import copy
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..validate import sha256


def _spec_list(config, name):
    values = config.get(name, [])
    return [values] if isinstance(values, dict) else values


def used_fit_specs(config, kind):
    """Only inputs consumed by this augmentation kind enter its resume hash."""
    if kind not in ("shadow_look", "camera_shadow"):
        return {}
    full = _spec_list(config, "full")
    names = [s["name"] for s in full]
    def selected(which, subset):
        by_name = {s["name"]: s for s in _spec_list(config, which)}
        return [{k: by_name[name][k] for k in ("name", "path", "sha256")}
                for name in subset if name in by_name]
    result = dict(libraries=names)
    if kind == "shadow_look":
        result["full"] = selected("full", names)
    else:
        result["wrist"] = selected("wrist", names)
        result["third"] = selected("third", names)
        supplied = {s["name"] for s in result["third"]}
        missing = [name for name in names if name not in supplied]
        if missing:
            result.update(full=selected("full", missing), store_root=config.get("store_root"), cell=config.get("cell"),
                          third_recipe="same-library PCA64 third + state; frozen A kwargs")
    return result


def _unwrap(method):
    seen = set()
    while id(method) not in seen:
        seen.add(id(method))
        if all(hasattr(method, k) for k in ("B0T", "B1T", "tasks", "cand_name", "query")):
            return method
        candidate = next((getattr(method, k) for k in ("base", "awm", "method") if hasattr(method, k)), None)
        if candidate is None:
            break
        method = candidate
    raise ValueError("artifact contains no fitted A retrieval")


def load_fit(spec):
    """Only trusted coordinator-provided fit artifacts are deserialized.

    Captured data remain pickle-free. Fits must have a declared content SHA.
    """
    path = Path(spec["path"])
    if not spec.get("sha256") or sha256(path) != spec["sha256"]:
        raise ValueError("fit SHA mismatch/missing for " + str(path))
    with path.open("rb") as f:
        blob = pickle.load(f)
    return _unwrap(blob["method"] if isinstance(blob, dict) else blob)


class _ThirdView:
    def __init__(self, query):
        self.query = query
    def __getattr__(self, name):
        return getattr(self.query, {"key_v1": "key_v0", "hist_key_v1": "hist_key_v0"}.get(name, name))


def make_third_fit(full_fit, root, cell, scratch):
    """Refit A's metric on PCA-64 third + state with its exact library rules.

    PCA uses the deployed library's existing third basis, not the wrist basis.
    The metric, early branch, kernel, state scale and continuity use A's fit
    recipe. No test observations or episode outcomes enter this fit.
    """
    from exp.offline_search.harness import api
    from exp.offline_search.rounds.r02.g1_awm.awm import AWM
    class ThirdAWM(AWM):
        def _feats(self, P0, P1, rs, which):
            return np.concatenate([P0, rs], axis=1).astype(np.float64)
        def fit(self, library, context):
            super().fit(library, context)
            self.B1T = np.empty((0, self.B0T.shape[1]), np.float32)
            self.muB1 = np.empty(0, np.float32)
            self.mu1 = self.mu0
            for task in self.tasks.values():
                task.V1, task.Vm1 = task.V0, task.Vm0
        def query(self, query):
            return super().query(_ThirdView(query))
    names = ("features", "lib", "fit_data", "kref", "k", "codes", "lam", "state_scale", "early",
             "step0_joint", "lam_c", "norm_cap", "hyst", "nn")
    kwargs = {k: getattr(full_fit, k) for k in names}
    if kwargs["fit_data"] != "same" or kwargs["features"] != "joint" or kwargs["step0_joint"]:
        raise ValueError("third-camera fit requires same-library joint A recipe")
    context = api.Context(root=root, cell=cell, seed=0, scratch=scratch)
    fit = ThirdAWM(**kwargs)
    fit.prof = api.NULL_PROFILER
    fit.fit(context.open_library("current"), context)
    return fit


class FrozenRetrieval:
    """One private fit per library; query history always follows factual actions.

    Config: full=[{path,sha256,name}], wrist=[...], optional third=[...],
    store_root, cell. Camera lists match the full library names. If third fits
    are absent they are fitted once by the same library-only A recipe.
    """
    def __init__(self, config, aug_dir, model, kinds=None):
        self.config = config
        self.model = model
        self.fits = {}
        self.provenance = {}
        full_specs = config.get("full", [])
        if not full_specs:
            raise ValueError("full frozen A fits required")
        if isinstance(full_specs, dict):
            full_specs = [full_specs]
        self.libraries = [s["name"] for s in full_specs]
        if len(set(self.libraries)) != len(self.libraries):
            raise ValueError("duplicate library in fit config")
        kinds = set(kinds if kinds is not None else (("shadow_look", "camera_shadow") if model == "pi05" else ("shadow_look",)))
        third_names = {s["name"] for s in _spec_list(config, "third")}
        full_names = set(self.libraries) if "shadow_look" in kinds else set(self.libraries) - third_names if "camera_shadow" in kinds else set()
        for kind, specs in (("full", full_specs), ("wrist", config.get("wrist", [])), ("third", config.get("third", []))):
            if isinstance(specs, dict):
                specs = [specs]
            for spec in specs:
                name = spec["name"]
                if name not in self.libraries or (kind == "full" and name not in full_names) or (kind != "full" and "camera_shadow" not in kinds):
                    continue
                fit = load_fit(spec)
                if fit.cand_name != name:
                    raise ValueError("fit library name mismatch: " + name)
                self.fits[kind, name] = fit
                self.provenance[kind + "/" + name] = dict(path=spec["path"], sha256=spec["sha256"])
        self.aug_dir = Path(aug_dir)

    def fit_specs(self, kind):
        return used_fit_specs(self.config, kind)

    def ensure_cameras(self):
        if self.model != "pi05":
            raise ValueError("camera shadows are supported for pi05 only")
        for name in self.libraries:
            if ("wrist", name) not in self.fits:
                raise ValueError("R7 frozen wrist fit required for " + name)
            if ("third", name) not in self.fits:
                config = self.config
                if "cell" not in config or "store_root" not in config:
                    raise ValueError("third-camera refit requires cell/store_root")
                fit = make_third_fit(self.fits["full", name], config["store_root"], config["cell"], self.aug_dir / "fit_scratch")
                self.fits["third", name] = fit
                digest = hashlib.sha256()
                for task in sorted(fit.tasks):
                    for attr in ("Wf", "shift", "W0f"):
                        value = getattr(fit.tasks[task], attr, None)
                        if value is not None:
                            digest.update(np.asarray(value).tobytes())
                self.provenance["third/" + name] = dict(recipe="same-library PCA64 third + state; frozen A kwargs", metric_sha256=digest.hexdigest())

    def retrieve(self, encoded, records, kind="full", history=None):
        from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
        history = history or {}
        if kind != "full":
            self.ensure_cameras()
        result = {}
        for library_i, name in enumerate(self.libraries):
            fit = self.fits[kind, name]
            rows, weights, scores, chunks, conf, d1 = [], [], [], [], [], []
            for i, record in enumerate(records):
                prefix = history.get(record["decision_id"], {})
                hist0 = prefix.get("v0", np.empty((0, encoded["v0"].shape[1]), np.float32))
                hist1 = prefix.get("v1", np.empty((0, encoded["v1"].shape[1]), np.float32))
                actions = prefix.get("served_chunk", np.empty((0, fit.H, 32), np.float32))
                ep = SimpleNamespace(uid=record["episode_key"], task_id=int(record["task_id"]), init=int(record["init"]))
                query = SimpleNamespace(key_v0=encoded["v0"][i], key_v1=encoded["v1"][i], rs=encoded["state"][i],
                                        raw_state=prefix.get("raw_state", encoded["state"][i]), task_id=ep.task_id,
                                        step=int(record["decision_seq"]), episode=ep,
                                        prev_hit=prefix.get("prev_hit"), prev_a_exec=actions[-1] if len(actions) else None,
                                        hist_key_v0=hist0, hist_key_v1=hist1, hist_a_exec=actions,
                                        hist_rs=prefix.get("state", np.empty((0, len(encoded["state"][i])))))
                # A's fitted arrays are immutable; copy only the few mutable
                # controller containers, reset and set factual gripper context.
                private = copy.copy(fit)
                private.reset(ep)
                if len(actions) and getattr(private, "hyst", 0):
                    private._last_g = float(np.sign(actions[-1, 4, 6]))
                response = private.query(query)
                row = np.asarray(response.topk, np.int32)
                score = np.asarray(response.scores, np.float64)
                w = _kernel_w(-score + score[0], private.kref)
                rows.append(row)
                weights.append((w / w.sum()).astype(np.float32))
                scores.append(score.astype(np.float32))
                chunks.append(np.asarray(response.action, np.float32))
                conf.append(response.confidence)
                d1.append((response.extras or {}).get("d1", np.nan))
            suffix = "" if library_i == 0 else "_" + name
            for key, values in (("rows", rows), ("weights", weights), ("scores", scores), ("cache_chunk", chunks),
                                ("conf", conf), ("d1", d1)):
                result[key + suffix] = np.asarray(values)
            result["lib" + suffix] = np.asarray([name] * len(records))
        if kind == "full":
            primary = self.fits["full", self.libraries[0]]
            result["keys_pca_third"] = (encoded["v0"] @ primary.B0T.T - primary.muB0).astype(np.float32)
            result["keys_pca_wrist"] = (encoded["v1"] @ primary.B1T.T - primary.muB1).astype(np.float32)
        return result


class FakeRetrieval:
    """CPU-only retrieval fixture; not an A-fit substitute in real jobs."""
    def __init__(self, model="pi05", libraries=("current",)):
        self.model, self.libraries = model, list(libraries)
        self.provenance = {"fixture": dict(recipe="fake-model tests only")}
    def fit_specs(self, kind):
        return dict(libraries=self.libraries, fixture=self.provenance)
    def ensure_cameras(self):
        if self.model != "pi05":
            raise ValueError("camera shadows pi05 only")
    def retrieve(self, encoded, records, kind="full", history=None):
        output = {}
        for j, name in enumerate(self.libraries):
            suffix = "" if j == 0 else "_" + name
            output.update({k + suffix: v for k, v in dict(rows=np.tile(np.arange(16, dtype=np.int32), (len(records), 1)),
                          weights=np.full((len(records), 16), 1 / 16., np.float32),
                          scores=np.tile(np.linspace(1, 0, 16, dtype=np.float32), (len(records), 1)),
                          cache_chunk=np.zeros((len(records), 10 if self.model == "pi05" else 16, 32), np.float32),
                          lib=np.array([name] * len(records))).items()})
        if kind == "full":
            output["keys_pca_third"] = encoded["v0"][:, :64]
            output["keys_pca_wrist"] = encoded["v1"][:, :64]
        return output
