"""Frozen Q3 progress-reliability recipe, independent of the call-budget solver.

``metric`` may be a fitted A/its wrapper or the explicit metric mapping documented
in INTERFACE.md. ``key`` may be {metric_code, metric}, an already encoded main
vector, or an A query carrying key_v*, rs and step. A mixed early/main window uses
each observation's own deployed metric against the corresponding template codes.
Only successful library episodes enter templates/calibration. No test trajectory,
success prediction, policy computation, randomization, or budget is used here.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np

SCHEMA = "r6.q3.stall.v1"


def _array(x, dtype=None):
    a = np.array(x, dtype=dtype, copy=True, order="C")
    a.flags.writeable = False
    return a


def _field(obj, name):
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def _manifest(library):
    return library["manifest"] if isinstance(library, Mapping) else library.meta


def _quantile(x, q):
    """The specified inverse ECDF; never NumPy's interpolated default."""
    a = np.sort(np.asarray(x, float))
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("quantile needs nonempty finite data")
    return float(a[max(0, math.ceil(q * len(a)) - 1)])


def _digest(obj):
    """Deterministic content hash, including array shape/dtype and numeric bytes."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, np.ndarray):
            h.update(b"array" + str(x.dtype).encode() + repr(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, Mapping):
            h.update(b"mapping")
            for k in sorted(x, key=lambda s: (type(s).__name__, str(s))):
                visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"sequence")
            for v in x: visit(v)
        else:
            h.update(json.dumps(x, sort_keys=True, allow_nan=False).encode() + b";")
    visit(obj)
    return h.hexdigest()


def monotone_alignment(distance):
    """Exact unrestricted nondecreasing DP with lexicographically first ties.

    Backward suffix optima allow a greedy earliest feasible row at each time.
    This handles ties between *whole paths*, not merely smallest predecessor.
    """
    d = np.asarray(distance, np.float64)
    if d.ndim != 2 or min(d.shape) < 1 or not np.isfinite(d).all():
        raise ValueError("alignment needs a finite, nonempty distance matrix")
    value = np.empty_like(d)
    value[-1] = d[-1]
    for t in range(len(d) - 2, -1, -1):
        value[t] = d[t] + np.minimum.accumulate(value[t+1, ::-1])[::-1]
    path = np.empty(len(d), np.int64)
    low = 0
    for t in range(len(d)):
        low += int(np.argmin(value[t, low:]))
        path[t] = low
    return float(value[0, path[0]] / len(d)), path


def _monotone_alignment_batch(distance, columns):
    """The same suffix DP, on (time, template, point) padded costs.

    Padding is +inf. Each addition has exactly the reference's two operands;
    only independent templates are batched. argmin's first occurrence keeps
    the lexicographically earliest entire path, including repeats and skips.
    Inputs are internal, already checked finite at the valid template points.
    """
    value = np.empty_like(distance)
    value[-1] = distance[-1]
    for t in range(len(distance) - 2, -1, -1):
        value[t] = distance[t] + np.minimum.accumulate(value[t+1, :, ::-1], axis=1)[:, ::-1]
    rows = np.arange(distance.shape[1])
    path = np.empty((len(distance), len(rows)), np.int64)
    path[0] = np.argmin(value[0], axis=1)
    for t in range(1, len(distance)):
        path[t] = np.argmin(np.where(columns >= path[t-1, :, None], value[t], np.inf), axis=1)
    return value[0, rows, path[0]] / len(distance), path


def _extract_metric(metric, manifest):
    if isinstance(metric, Mapping):
        # Protocol for another benchmark/robot: provide codes + optional encoder.
        out = dict(metric)
        if "tasks" not in out:
            raise ValueError("metric mapping requires tasks")
        out.setdefault("projection", None)
        out.setdefault("provenance", {})
        return out
    awm = metric
    for _ in range(16):
        if hasattr(awm, "tasks") and hasattr(awm, "_dist"): break
        awm = getattr(awm, "base", None)
        if awm is None: raise TypeError("metric must be fitted A or an explicit mapping")
    else:
        raise TypeError("cannot locate fitted A metric")
    cameras = sorted(int(m[1]) for k in vars(awm)
                     if (m := re.fullmatch(r"B(\d+)T", k)))
    if not cameras: raise ValueError("A fit has no fitted visual projections")
    projection = dict(cameras={}, tasks={}, state_dims=int(manifest["rs_valid_dims"]),
                      main_features=awm.features, early_features=awm.feat0,
                      early=bool(awm.early))
    for c in cameras:
        projection["cameras"][str(c)] = dict(B=_array(getattr(awm, f"B{c}T")),
                                             mu=_array(getattr(awm, f"muB{c}")))
    tasks = {}
    for key, t in sorted(awm.tasks.items()):
        codes = {"main": _array(t.Z)}
        if awm.early:
            early = t.Z0
            if early is None:
                early = t.Z @ t.A0
                if t.As0 is not None: early = early + t.RS @ t.As0
            codes["early"] = _array(early)
        tasks[str(key)] = dict(rows=_array(t.rows, np.int64), codes=codes)
        projection["tasks"][str(key)] = dict(W=_array(t.Wf), shift=_array(t.shift),
            W0=_array(t.W0f) if awm.early else None, shift0=_array(t.c0) if awm.early else None)
    return dict(tasks=tasks, projection=projection,
                provenance=dict(kind="deployed_A_frozen_representation_and_metric",
                                fit_content_sha256=_digest(dict(tasks=tasks, projection=projection))))


def _cdf_table(values, episodes):
    """Every calibration episode has total mass one, divided over its windows."""
    values, episodes = np.asarray(values), np.asarray(episodes)
    weights = np.empty(len(values), float)
    for ep in np.unique(episodes):
        where = episodes == ep
        weights[where] = 1. / np.count_nonzero(where)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return dict(values=_array(values[order]), cumulative=_array(cumulative))


def _cdf(table, value):
    pos = int(np.searchsorted(table["values"], value, side="right"))
    return 0. if pos == 0 else float(table["cumulative"][pos-1])


class StallModel:
    """Immutable-by-contract library artifact; one tracker per live episode."""

    @classmethod
    def fit(cls, library, *, metric, commit_controls):
        manifest = _manifest(library)
        H, R = int(manifest["H"]), int(manifest["exec_steps"])
        if H <= 0 or R <= 0 or isinstance(commit_controls, bool) or commit_controls != min(H, 2*R):
            raise ValueError("commit_controls must equal min(manifest H, 2*exec_steps)")
        L = int(commit_controls)
        model = cls()
        model.commit_controls, model.exec_steps = L, R
        model.task_map = {str(k): str(v) for k, v in manifest.get("task_map", {}).items()}
        model.metric = _extract_metric(metric, manifest)
        task, ep, step, success = [np.asarray(_field(library, k)) for k in
                                   ("task_id", "episode", "step", "success")]
        if not (task.shape == ep.shape == step.shape == success.shape):
            raise ValueError("library row labels differ in shape")
        # The legacy store's step is explicitly in exec_steps units. A portable
        # input may instead supply actual control timestamps; no robot constant.
        if isinstance(library, Mapping) and "control_index" in library:
            controls = np.asarray(library["control_index"], np.int64)
        else:
            controls = np.asarray(step, np.int64) * R
        if controls.shape != step.shape or not np.isfinite(controls).all():
            raise ValueError("invalid library control timestamps")
        model.provenance = dict(schema=SCHEMA, manifest_sha256=_digest(manifest),
            bank_labels_sha256=_digest(dict(task=task, episode=ep, step=step, controls=controls, success=success)),
            metric_sha256=_digest(model.metric), metric_provenance=model.metric.get("provenance", {}),
            constants=dict(window_fraction=.05, minimum_window_intervals=2, max_templates=5,
                           error_quantile=.90, advance_quantile=.10, minimum_reference_episodes=4),
            calibration_scope="successful library; frozen representation; empirical not conformal; no evaluation data")
        model.tasks = {}
        for task_key, mt in sorted(model.metric["tasks"].items(), key=lambda x: str(x[0])):
            key = str(task_key)
            rows = np.asarray(mt["rows"], np.int64)
            if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(task)):
                raise ValueError("metric has invalid candidate row identities")
            if any(str(x) != key for x in task[rows]):
                raise ValueError("metric task and library row task mismatch")
            codes = {m: np.asarray(a) for m, a in mt["codes"].items()}
            if "main" not in codes or any(a.ndim != 2 or len(a) != len(rows) or not np.isfinite(a).all() for a in codes.values()):
                raise ValueError("invalid metric codes")
            templates = []
            for episode in sorted(np.unique(ep[rows]).tolist()):
                positions = np.flatnonzero(ep[rows] == episode)
                positions = positions[np.argsort(controls[rows[positions]], kind="stable")]
                rr = rows[positions]
                if not np.all(success[rr]):
                    if np.any(success[rr]): raise ValueError("inconsistent episode success metadata")
                    continue
                ts = controls[rr]
                if len(ts) < 2 or ts[-1] <= ts[0]: continue
                if np.any(np.diff(ts) <= 0): raise ValueError("episode timestamps must increase")
                ts = ts - ts[0]
                grid = np.r_[np.arange(0, ts[-1] + 1, L, dtype=np.int64), ts[-1]]
                take = np.unique(np.searchsorted(ts, grid, side="right") - 1)
                pos, tt = positions[take], ts[take]
                templates.append(dict(episode=episode, rows=_array(rows[pos]), controls=_array(tt),
                    phase=_array(tt / float(ts[-1])), duration=int(ts[-1]),
                    codes={m:_array(a[pos], np.float64) for m, a in codes.items()}))
            E = len(templates)
            median_intervals = float(np.median([p["duration"] / L for p in templates])) if E else 0.
            W = max(2, math.ceil(.05 * median_intervals))
            data = dict(E=E, K=min(5, max(E-1, 0)), W=W, T_med=median_intervals,
                        templates=templates, calibration=[], references=[], cdfs=None)
            model.tasks[key] = data
            if E < 2: continue
            records = []
            for template in templates:
                for end in range(W, len(template["controls"])):
                    start = end-W
                    window=[]
                    for j in range(start, end+1):
                        regime = "early" if j == 0 and "early" in template["codes"] else "main"
                        window.append((template["codes"][regime][j], regime))
                    span = int(template["controls"][end] - template["controls"][start])
                    pred = model._estimate(key, window, span, exclude=template["episode"])
                    truth = float((template["phase"][end] - template["phase"][start]) * W*L/span)
                    records.append(dict(episode=template["episode"], end_control=int(template["controls"][end]),
                        window_span=span, delta_true=truth, residual=truth-pred["delta_hat"], **pred))
            data["calibration"] = records
            if not records: continue
            episodes = np.array([r["episode"] for r in records])
            cdfs = {name:_cdf_table([r[name] for r in records], episodes) for name in ("distance_hat", "spread_hat")}
            data["cdfs"] = cdfs
            for episode in sorted(np.unique(episodes).tolist()):
                rr = [r for r in records if r["episode"] == episode]
                data["references"].append(dict(episode=episode,
                    context=_array([model._context(key,r) for r in rr]),
                    residual=_array([r["residual"] for r in rr]),
                    advance=_array([r["delta_true"] for r in rr]),
                    end_control=_array([r["end_control"] for r in rr], np.int64)))
        model.fingerprint = _digest(model._payload())
        return model

    def _payload(self):
        return dict(schema=SCHEMA, commit_controls=self.commit_controls, exec_steps=self.exec_steps,
                    task_map=self.task_map, metric=self.metric, tasks=self.tasks, provenance=self.provenance)

    def resolve_task(self, task_key):
        key = str(task_key)
        if key in self.tasks: return key
        if key in self.task_map and self.task_map[key] in self.tasks: return self.task_map[key]
        return None

    def encode(self, key, task_key):
        """Reuse deployed codes, or project a raw QueryView without querying A."""
        if isinstance(key, Mapping) and ("metric_code" in key or "code" in key):
            code = np.asarray(key.get("metric_code", key.get("code")), float)
            regime = key.get("metric", key.get("regime", "main"))
        elif isinstance(key, (np.ndarray, list, tuple)):
            code, regime = np.asarray(key, float), "main"
        else:
            p = self.metric.get("projection")
            if p is None: raise ValueError("raw key requires a saved deployed projection")
            xv = np.concatenate([c["B"] @ np.asarray(_field(key,f"key_v{camera}"),np.float32)-c["mu"]
                                 for camera,c in p["cameras"].items()])
            regime = "early" if p["early"] and int(_field(key,"step")) == 0 else "main"
            feat = p["early_features"] if regime == "early" else p["main_features"]
            x = np.concatenate([xv,np.asarray(_field(key,"rs"),np.float32)[:p["state_dims"]]]) if feat == "joint" else xv
            t = p["tasks"][task_key]
            code = x @ t["W0"]-t["shift0"] if regime == "early" else x @ t["W"]-t["shift"]
        if regime not in ("main", "early") or code.ndim != 1 or not np.isfinite(code).all():
            raise ValueError("invalid metric code or regime")
        if self.tasks[task_key]["templates"]:
            example = self.tasks[task_key]["templates"][0]["codes"]
            if regime not in example or code.shape != (example[regime].shape[1],):
                raise ValueError("code does not match the saved metric regime/dimension")
        return _array(code,np.float64),regime

    def _template_cache(self, task_key):
        """Derived arrays stay outside tasks/_payload; never write them to disk."""
        if not hasattr(self, '_template_caches'):
            self._template_caches = {}
        if task_key not in self._template_caches:
            templates = self.tasks[task_key]['templates']
            lengths = np.array([len(t['phase']) for t in templates], np.int64)
            n = int(max(lengths, default=0))
            columns = np.arange(n)
            valid = columns < lengths[:, None]
            phase = np.zeros(valid.shape)
            for i, t in enumerate(templates):
                phase[i, :lengths[i]] = t['phase']
            # Flatten valid rows for the metric arithmetic: same contiguous
            # float64 ij,ij->i einsum as before, without computing padding.
            codes = {regime: np.concatenate([t['codes'][regime] for t in templates])
                     for regime in templates[0]['codes']} if templates else {}
            self._template_caches[task_key] = dict(codes=codes, phase=phase,
                valid=valid, columns=columns, rows=np.arange(len(templates)),
                episodes=np.array([t['episode'] for t in templates]))
        return self._template_caches[task_key]

    def _distances(self, task_key, encoded):
        cache = self._template_cache(task_key)
        z, regime = encoded
        difference = cache['codes'][regime] - z
        flat = np.sqrt(np.einsum('ij,ij->i', difference, difference))
        if not np.isfinite(flat).all():
            raise ValueError('alignment needs a finite, nonempty distance matrix')
        distances = np.full(cache['valid'].shape, np.inf)
        distances[cache['valid']] = flat
        return distances

    def _estimate(self, task_key, window, span, exclude=None):
        return self._estimate_distances(task_key,
            np.asarray([self._distances(task_key, v) for v in window]), span, exclude)

    def _estimate_distances(self, task_key, distances, span, exclude=None, *, details=False):
        data = self.tasks[task_key]
        cache = self._template_cache(task_key)
        if not len(cache['rows']) or not data['K']:
            raise ValueError('insufficient alignment templates')
        means, paths = _monotone_alignment_batch(distances, cache['columns'])
        # Stable sorting is exactly (mean, original template order).
        eligible = cache['rows'][cache['episodes'] != exclude]
        best = eligible[np.argsort(means[eligible], kind='stable')[:data['K']]]
        if len(best) != data['K'] or not len(best):
            raise ValueError('insufficient alignment templates')
        phase = cache['phase']
        starts = phase[best, paths[0, best]]
        # Keep subtraction/multiplication/division in the original order.
        advances = (phase[best, paths[-1, best]] - starts) * data['W'] * self.commit_controls / span
        sorted_advances = np.sort(advances)
        k = len(best)
        middle = max(0, math.ceil(.5*k)-1)
        estimate = dict(delta_hat=float(sorted_advances[middle]),
            phase_hat=float(np.sort(starts)[middle]), distance_hat=float(np.sort(means[best])[middle]),
            spread_hat=float(sorted_advances[max(0, math.ceil(.9*k)-1)]) -
                       float(sorted_advances[max(0, math.ceil(.1*k)-1)]))
        if details:
            return estimate, best, paths
        return estimate

    def _estimate_reference(self, task_key, window, span, exclude=None):
        from .stall_reference import StallModel as ReferenceModel
        return ReferenceModel._estimate(self, task_key, window, span, exclude)

    def _context(self, task_key, estimate):
        c=self.tasks[task_key]["cdfs"]
        return np.array([estimate["phase_hat"],_cdf(c["distance_hat"],estimate["distance_hat"]),
                         _cdf(c["spread_hat"],estimate["spread_hat"])])

    def _reference_cache(self, task_key):
        if not hasattr(self, '_reference_caches'):
            self._reference_caches = {}
        if task_key not in self._reference_caches:
            refs = self.tasks[task_key]['references']
            lengths = np.array([len(r['context']) for r in refs], np.int64)
            valid = np.arange(int(max(lengths, default=0))) < lengths[:, None]
            context = np.zeros((*valid.shape, 3))
            residual, advance = np.zeros(valid.shape), np.zeros(valid.shape)
            end = np.zeros(valid.shape, np.int64)
            for i, r in enumerate(refs):
                context[i, :lengths[i]] = r['context']
                residual[i, :lengths[i]] = r['residual']
                advance[i, :lengths[i]] = r['advance']
                end[i, :lengths[i]] = r['end_control']
            self._reference_caches[task_key] = dict(context=context, valid=valid,
                residual=residual, advance=advance, end=end,
                episodes=[r['episode'] for r in refs], rows=np.arange(len(refs)))
        return self._reference_caches[task_key]

    def calibrated_status(self, task_key, estimate, span):
        data = self.tasks[task_key]
        cache = self._reference_cache(task_key)
        # Three-component sums use the same contiguous last axis as the
        # reference's (windows, 3) arrays; argmin keeps earliest window ties.
        distance = np.abs(cache['context'] - self._context(task_key, estimate)).sum(axis=2)
        distance[~cache['valid']] = np.inf
        j = np.argmin(distance, axis=1)
        rows = cache['rows']
        residuals = cache['residual'][rows, j]
        advances = cache['advance'][rows, j]
        e90, a10 = _quantile(residuals, .9), _quantile(advances, .1)
        selected = [[ep, int(end)] for ep, end in zip(cache['episodes'], cache['end'][rows, j], strict=True)]
        delta = estimate['delta_hat']
        state = 'slow_confirmed' if delta+e90<a10 else 'slow_ambiguous' if delta<a10<=delta+e90 else 'ok'
        return dict(state=state, **estimate, e90=e90, a10=a10, window_span=int(span),
                    W=data['W'], K=data['K'], reference_episodes=len(residuals), reference_windows=selected)

    def calibrated_status_reference(self, task_key, estimate, span):
        from .stall_reference import StallModel as ReferenceModel
        return ReferenceModel.calibrated_status(self, task_key, estimate, span)

    def save(self,path):
        """Save a self-contained trusted artifact directory, or a .pkl path."""
        path=Path(path)
        file=path if path.suffix==".pkl" else path/"stall.pkl"
        file.parent.mkdir(parents=True,exist_ok=True)
        payload=self._payload()
        fingerprint=_digest(payload)
        if fingerprint!=self.fingerprint: raise ValueError("model was modified after fit/load")
        raw=pickle.dumps(payload,protocol=4)
        sidecar=file.with_suffix('.json')
        if file.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to overwrite artifact {file}")
        file.write_bytes(raw)
        meta=dict(schema=SCHEMA,fingerprint=fingerprint,payload_sha256=hashlib.sha256(raw).hexdigest(),
                  provenance=self.provenance,commit_controls=self.commit_controls,exec_steps=self.exec_steps,
                  tasks={k:dict(E=v["E"],K=v["K"],W=v["W"],T_med=v["T_med"],
                      calibration_windows=len(v["calibration"]),reference_episodes=len(v["references"]),
                      available=len(v["references"])>=4) for k,v in self.tasks.items()})
        sidecar.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        with file.with_suffix('.loeo.csv').open('w',newline='') as f:
            fields=['task_key','episode','end_control','window_span','delta_true','residual','delta_hat','phase_hat','distance_hat','spread_hat']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for task_key,data in self.tasks.items():
                for record in data['calibration']:writer.writerow(dict(task_key=task_key,**record))

    @classmethod
    def load(cls,path):
        path=Path(path);file=path if path.suffix=='.pkl' else path/'stall.pkl'
        meta=json.loads(file.with_suffix('.json').read_text());raw=file.read_bytes()
        if meta.get('schema')!=SCHEMA or hashlib.sha256(raw).hexdigest()!=meta['payload_sha256']:
            raise ValueError('artifact schema/hash mismatch')
        # Only load trusted locally built artifacts: pickle is not a sandbox.
        payload=pickle.loads(raw)
        if _digest(payload)!=meta['fingerprint']:raise ValueError('artifact content fingerprint mismatch')
        obj=cls()
        for k,v in payload.items():
            if k!='schema':setattr(obj,k,v)
        obj.fingerprint=meta['fingerprint']
        return obj


class StallTracker:
    """One episode. observe does the work; status only returns cached diagnostics.

    No MISS/LOOK is executed here. C applies its shared rho budget, cooldown,
    and one-extra-LOOK-per-W*L control window to the returned status.
    """
    def __init__(self,model:StallModel,task_key):
        self.model=model;self.task_key=model.resolve_task(task_key)
        data=model.tasks.get(self.task_key)
        self._window=deque(maxlen=data['W']+1 if data else 1)
        self._distance_window=deque(maxlen=self._window.maxlen)
        self._last_control=None
        self._status=self._inactive('unobserved')

    def _inactive(self,reason):
        d=self.model.tasks.get(self.task_key,{})
        return dict(state='inactive',reason=reason,delta_hat=None,e90=None,a10=None,window_span=0,
                    W=d.get('W'),K=d.get('K'),reference_episodes=len(d.get('references',[])))

    def observe(self,key,control_index:int)->None:
        if isinstance(control_index,bool) or not isinstance(control_index,(int,np.integer)) or control_index<0:
            raise ValueError('control_index must be a nonnegative integer')
        control_index=int(control_index)
        if self._last_control is not None and control_index<=self._last_control:
            raise ValueError('fresh observations must have strictly increasing control indices; use one tracker per episode')
        self._last_control=control_index
        if self.task_key is None:
            self._status=self._inactive('unknown_task');return
        d=self.model.tasks[self.task_key]
        if len(d['references'])<4:
            self._status=self._inactive('fewer_than_four_reference_episodes');return
        try:
            encoded=self.model.encode(key,self.task_key)
        except (ValueError,KeyError,TypeError,AttributeError):
            # A bad observation breaks a window; do not bridge through missing
            # vision or silently treat a malformed metric code as zero motion.
            self._window.clear();self._distance_window.clear()
            self._status=self._inactive('invalid_observation');return
        self._window.append((encoded,control_index))
        if len(self._window)<d['W']+1:
            self._status=self._inactive('warming_up');return
        span=self._window[-1][1]-self._window[0][1]
        # Defer distance arithmetic until the first complete window, preserving
        # warmup/error behavior even for finite codes whose squared norm overflows.
        try:
            if not self._distance_window:
                distances = [self.model._distances(self.task_key, v[0]) for v in self._window]
                self._distance_window.extend(distances)
            else:
                self._distance_window.append(self.model._distances(self.task_key, encoded))
        except ValueError:
            self._distance_window.clear()
            raise
        estimate=self.model._estimate_distances(self.task_key,np.asarray(self._distance_window),span)
        self._status=self.model.calibrated_status(self.task_key,estimate,span)

    def status(self)->dict:
        out=dict(self._status)
        if 'reference_windows' in out:out['reference_windows']=[list(v) for v in out['reference_windows']]
        return out
