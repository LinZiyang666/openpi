"""Passive capture at request entry and after the live response is fixed."""
import contextlib
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np

from .. import schema
from .diagnostics import debug_record, detached, metric_tap, split_diagnostics
from .manifest import model_manifest, startup_meta
from .digests import DigestCache, cache_path
from .writer import BlockWriter, atomic_json

log = logging.getLogger("osdebug")


class UnavailableObserver:
    """Fail-open telemetry when startup storage/provenance cannot be written."""
    def __init__(self, runtime, error):
        self.runtime, self.error = runtime, str(error)
        self._seq = 0
        self._lock = threading.Lock()

    def begin(self, obs, envelope):
        class Capture:
            def observe(self, session):
                return contextlib.nullcontext()
        capture = Capture()
        capture.envelope = dict(envelope) if isinstance(envelope, dict) else {}
        capture.dispatch, capture.stage_ms = {}, {}
        return capture

    def finish(self, capture, session, response, infer_ms, ok=True, error=None):
        with self._lock:
            seq = self._seq
            self._seq += 1
        did = capture.envelope.get("decision_id")
        log.error("osdebug unavailable for %s: %s", did, self.error)
        return dict(v=1, decision_id=did, server_tag=self.runtime.tag, server_seq=seq, status="error")

    def close(self):
        pass


def parse_config(text):
    value = json.loads(text or "{}")
    if not isinstance(value, dict):
        raise ValueError("--os-debug-config must be a JSON object")
    if int(value.get("writer_queue_bytes", value.get("queue_bytes", 512 * 1024 ** 2))) <= 0:
        raise ValueError("writer queue bytes must be positive")
    # Sampling is the shared hash design, never serving RNG. Accept its explicit
    # rates, but reject incompatible rates rather than silently changing design.
    for key, rate in (("rawkeys_rate", 1 / 16), ("snapshot_rate", 1 / 16), ("draws_rate", 1 / 32)):
        if key in value and float(value[key]) != rate:
            raise ValueError(key + " is fixed by osdebug.v1")
    return value


class RequestCapture:
    def __init__(self, observer, obs, envelope):
        self.observer = observer
        self.t_recv = time.time()
        self.started = time.perf_counter()
        self.envelope = dict(envelope) if isinstance(envelope, dict) else {}
        self.arrays = {}
        self.metric = {}
        self.errors = []
        self.prompt = ""
        self.before_calls = {}
        self.dispatch = {}
        self.stage_ms = {}
        try:
            manifest = observer.manifest
            self.arrays["state_wire"] = detached(obs["observation/state"], np.float64).reshape(-1)
            for camera, wire in manifest["camera_wire_keys"].items():
                array = detached(obs[wire])
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                    raise ValueError("wire image must be uint8 H,W,3: " + camera)
                self.arrays["img_" + camera] = array
            self.prompt = str(obs.get("prompt", ""))
        except Exception as exc:
            self.errors.append("wire capture: {}: {}".format(type(exc).__name__, exc))

    @contextlib.contextmanager
    def observe(self, session):
        self.before_calls[id(session)] = getattr(session, "stage1_calls", 0)
        with metric_tap(session, self.metric):
            yield


class ServerObserver:
    def __init__(self, runtime, directory, config=None):
        self.runtime = runtime
        self.directory = Path(directory)
        self.config = dict(config or {})
        for name, value in (("rawkeys_rate", 1 / 16), ("snapshot_rate", 1 / 16), ("draws_rate", 1 / 32),
                            ("sampling_algorithm", "sha256(campaign|task_uid|decision_seq), big-endian integer")):
            self.config.setdefault(name, value)
        self.campaign = str(self.config.get("campaign", self.directory.parent.parent.parent.parent.name))
        self.manifest = model_manifest(runtime, self.config)
        self._seq = 0
        self._lock = threading.Lock()
        self._anchors = {}
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = startup_meta(runtime, self.config, self.manifest, DigestCache(cache_path(self.directory)))
        self.runtime.debug_library_shas = {}
        content_fingerprints = {}
        for lib in getattr(runtime, "tables", {}):
            entries = {k: v for k, v in meta["artifacts"].items()
                       if k == "library_" + lib or k.startswith("library_" + lib + "_")}
            import hashlib
            digests = {k: v.get("sha256", v) if isinstance(v, dict) else v for k, v in entries.items()}
            content_fingerprints[lib] = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
            library_meta = meta["library_meta"] if lib == "current" else {}
            manifest_artifact = entries.get("library_" + lib + "_manifest", {})
            if manifest_artifact.get("status") == "available":
                library_meta = json.loads(Path(manifest_artifact["path"]).read_text())
            self.runtime.debug_library_shas[lib] = (library_meta.get("sources") or {}).get("pkl_sha256", content_fingerprints[lib])
        meta["library_shas"] = self.runtime.debug_library_shas
        meta["library_content_fingerprints"] = content_fingerprints
        self.meta_path = self.directory / "meta_{}.json".format(os.getpid())
        atomic_json(self.meta_path, meta)
        self.writer = BlockWriter(self.directory, self.config.get("writer_queue_bytes",
                                  self.config.get("queue_bytes", 512 * 1024 ** 2)))
        self._old_sigterm = None
        # Only debug-on installs a handler. Drain before delegating to the stock
        # handler, or exit normally when SIGTERM had its default disposition.
        if threading.current_thread() is threading.main_thread():
            self._old_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._sigterm)

    def _sigterm(self, signum, frame):
        self.close()
        previous = self._old_sigterm
        if callable(previous):
            previous(signum, frame)
        elif previous != signal.SIG_IGN:
            raise SystemExit(128 + signum)

    def close(self):
        self.writer.close()
        if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) == self._sigterm:
            signal.signal(signal.SIGTERM, self._old_sigterm)

    def flush(self):
        """Make all admitted requests durable before an episode can be accepted."""
        self.writer.flush()
        if self.writer.error:
            raise RuntimeError("debug episode capture failed: " + self.writer.error)

    def begin(self, obs, envelope):
        return RequestCapture(self, obs, envelope)

    def _identity(self, capture, session, seq):
        env = capture.envelope
        meta = getattr(session, "ep_meta", {}) or {}
        uid = str(env.get("task_uid", meta.get("uid", "unverified:c{}".format(getattr(session, "conn", 0)))))
        attempt = int(env.get("attempt", meta.get("attempt", 1)))
        epkey = schema.episode_key(uid, attempt)
        generation = int(env.get("dispatch_gen", 0))
        dseq = int(env.get("decision_seq", max(0, getattr(session, "step", seq + 1) - 1)))
        did = schema.decision_id(epkey, generation, dseq)
        required = {"v", "task_uid", "attempt", "episode_key", "dispatch_gen", "decision_seq", "decision_id", "t_client_send"}
        if (not required.issubset(env) or env.get("v") != 1 or env.get("episode_key") != epkey
                or env.get("decision_id") != did):
            capture.errors.append("missing or inconsistent debug identity envelope")
        # Preserve received ID even on mismatch so client can join the error.
        try:
            init = int(uid.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            init = None
            capture.errors.append("task_uid has no subset init index")
        return dict(decision_id=str(env.get("decision_id", did)), episode_key=str(env.get("episode_key", epkey)),
                    task_uid=uid, attempt=attempt, decision_seq=dseq, dispatch_gen=generation,
                    task_id=meta.get("task_id"), init=init,
                    orig_init_state_idx=meta.get("orig_init_state_idx", meta.get("init")),
                    t_client_send=env.get("t_client_send"))

    def finish(self, capture, session, response, infer_ms, ok=True, error=None):
        """All failures are telemetry errors; the caller's action stays fixed."""
        with self._lock:
            seq = self._seq
            self._seq += 1
        record = dict(schema=schema.SCHEMA_VERSION, server_tag=getattr(self.runtime, "tag", "p" + str(os.getpid())),
                      server_seq=seq, conn=getattr(session, "conn", None), pid=os.getpid(), t_recv=capture.t_recv,
                      t_done=time.time(), infer_ms=infer_ms, ok=bool(ok))
        arrays, raw = capture.arrays, {}
        try:
            record.update(self._identity(capture, session, seq))
            self._decision(capture, session, response, record, arrays, raw)
        except Exception as exc:
            capture.errors.append("decision capture: {}: {}".format(type(exc).__name__, exc))
            # Keep a joinable error record even if the envelope is malformed.
            for key in ("decision_id", "episode_key", "decision_seq", "dispatch_gen", "task_uid", "attempt"):
                record.setdefault(key, capture.envelope.get(key, "unverified:{}".format(seq)))
        if error:
            record["serving_error"] = str(error)
        record["capture_errors"] = capture.errors
        record["status"] = "error" if capture.errors or self.writer.error or not ok else "available"
        record["observer_ms"] = (time.perf_counter() - capture.started) * 1000 - infer_ms
        try:
            record = split_diagnostics(record, arrays, "diag_record")
            self.writer.put(record, arrays, raw, capture.prompt)
        except Exception as exc:
            record["status"] = "error"
            log.error("osdebug capture failed for %s: %s", record["decision_id"], exc)
        return dict(v=1, decision_id=record["decision_id"], server_tag=record["server_tag"],
                    server_seq=seq, status=record["status"])

    def _decision(self, capture, session, response, record, arrays, raw):
        d = getattr(session, "_dec", None) or {}
        vision = bool(d.get("vision", bool(d)))
        hit = bool(d.get("hit", True))
        source = d.get("source", "cache" if hit else "policy")
        if source == "cache_blind":
            source = "cache_tail"
        method = getattr(session, "method", None)
        camera = getattr(session, "_camera_mode", self.config.get("camera_mode", "full")) if vision else "blind"
        lookstep = int(getattr(session, "last_vision_step", 0))
        step = int(d.get("step", max(0, getattr(session, "step", 1) - 1)))
        controls = int(self.manifest["block_controls"])
        age = controls * max(0, step - lookstep) if not vision else 0
        anchor_key = (getattr(session, "conn", 0), record["episode_key"], record["dispatch_gen"])
        if vision:
            self._anchors[anchor_key] = (record["decision_id"], step)
        anchor, anchor_step = self._anchors.get(anchor_key, (record["decision_id"], step))
        if source in ("cache", "cache_tail") and not vision:
            extras = d.get("extras") or {}
            source = "follow" if "os_sf_source" in extras or extras.get("os_sf_extension") == 1 else "cache_tail"
        work = getattr(session, "_camera_work", {}) if vision else {}
        calls = int(getattr(session, "stage1_calls", 0) - capture.before_calls.get(id(session), 0))
        if "stage1" in capture.dispatch:
            calls = capture.dispatch["stage1"]
        # Non-R4 CP1 requests consume exactly one stage-1 result. R4 uses its
        # actual preexisting invocation counter. No diagnostic dispatch exists.
        if vision and not getattr(self.runtime, "r4", False):
            calls = 1
        completions = int(work.get("completions", 0))
        policy_calls = int(vision and not hit)
        prices = self.manifest["cost_weights"]
        owner_cost = calls * prices.get(camera, prices.get("full", 0.)) + completions * prices.get("completion", 0.)
        owner_cost += policy_calls * prices.get("policy", 0.)
        s23 = (d["t_exec"] - d["t_s1"]) / 1e6 if policy_calls and d.get("t_exec") else None
        pre = (d["t_s0"] - session.t_obs) / 1e6 if d.get("t_s0") and getattr(session, "t_obs", None) else None
        record.update(vision=vision, camera_mode=camera, src=source, hit=hit,
                      blind_age_controls=age, anchor_decision_id=anchor,
                      chunk_offset=controls * max(0, step - anchor_step),
                      look_reason=getattr(session, "_look_reason", None), miss_reason=d.get("judge") if not hit else None,
                      stage1_calls=calls, camera_completions=completions, policy_calls=policy_calls,
                      stage23_calls=policy_calls, owner_cost=owner_cost,
                      pre_ms=pre, s1_ms=getattr(session, "_s1_ms", None) if vision else None,
                      s23_ms=s23, method_ms=float(d.get("q_us", 0)) / 1000,
                      decision_index=getattr(session, "_decision_index", step),
                      timings_status=schema.status("available" if getattr(self.runtime, "r4", False) else "unsupported",
                                                   "stage timing hooks absent on legacy non-R4 path"),
                      oracle_status=schema.status("available" if getattr(self.runtime, "oracle", False) else "not_applicable",
                                                  "privileged diagnostic arm" if getattr(self.runtime, "oracle", False) else "oracle flag absent"))
        dispatch_status = getattr(session, "_debug_dispatch_status", schema.status("unsupported", "no dispatch tap"))
        record["dispatch_status"] = dispatch_status
        record["stage_invocations"] = dict(capture.dispatch)
        record["stage_dispatch_ms"] = dict(capture.stage_ms)
        if dispatch_status["status"] == "available":
            record["stage23_calls"] = max(capture.dispatch.get("stage23", 0), capture.dispatch.get("stage2", 0),
                                          capture.dispatch.get("stage3", 0))
            record["s1_ms"] = capture.stage_ms.get("stage1", 0.) if vision else None
            record["s23_ms"] = (capture.stage_ms["stage23"] if "stage23" in capture.stage_ms else
                                 capture.stage_ms.get("stage2", 0.) + capture.stage_ms.get("stage3", 0.)) if policy_calls else None
            record["timings_status"] = schema.status("available", "live stage wall times include coordinator waits")
        oracle = getattr(session, "_oracle", None)
        if oracle is not None:
            record["oracle"] = split_diagnostics(oracle, arrays, "diag_oracle")
        oracle_error = getattr(session, "_oracle_error", None)
        if oracle_error:
            record["oracle_status"] = schema.status("error", oracle_error)
            capture.errors.append(oracle_error)
        if response is not None and "actions" in response:
            arrays["served_wire"] = detached(response["actions"], np.float64)  # lossless: wire actions are float64
            record["served_len"] = len(arrays["served_wire"])
        else:
            record["served_wire_status"] = schema.status("error", "no successful wire response")
            arrays["served_wire"] = np.full((int(self.manifest["H"]), len(self.manifest["valid_action_dims"])), np.nan, np.float64)
        served = d.get("served") if hit else d.get("policy")
        if served is not None:
            arrays["served_chunk"] = detached(served, np.float32)
        record["served_chunk_status"] = schema.status("available" if served is not None else "unsupported",
                                                     "normalized chunk unavailable" if served is None else "")
        for key, value in (("cache_chunk", d.get("served") if vision else None), ("policy_chunk", d.get("policy"))):
            if value is not None:
                arrays[key] = detached(value, np.float32)
            record[key + "_status"] = schema.status("available" if value is not None else "not_applicable",
                                                     "not computed live" if value is None else "")
        for key in ("served_chunk", "cache_chunk", "policy_chunk"):
            arrays.setdefault(key, np.full((int(self.manifest["H"]), int(self.manifest["action_dim"])), np.nan, np.float32))
        buf = getattr(session, "b_rs", None)
        if buf is not None and buf.n > step:
            arrays["state_norm"] = detached(buf.a[step], np.float32)
        record["state_norm_status"] = schema.status("available" if "state_norm" in arrays else "unsupported",
                                                   "no captured normalized input" if "state_norm" not in arrays else "")
        arrays.setdefault("state_norm", np.full(int(self.manifest["normalized_state_dim"]), np.nan, np.float32))
        keys_status = {}
        for camera_name in self.manifest["camera_names"]:
            name = "keys_pca_" + camera_name
            value = capture.metric.get(name)
            if value is not None:
                arrays[name] = value
            else:
                arrays[name] = np.full(64, np.nan, np.float32)
            keys_status[camera_name] = schema.status("available" if value is not None else "not_applicable" if not vision
                                                    or camera == "wrist_only" and camera_name == "third" else "unsupported",
                                                    "no live compact key" if value is None else "")
        record["keys_status"] = keys_status
        if capture.metric.get("tap_error"):
            capture.errors.append(capture.metric["tap_error"])
        ex = d.get("extras") or {}
        record["extras"] = split_diagnostics(ex, arrays, "diag_extras")
        record["blind_extras"] = split_diagnostics(getattr(session, "_blind_extras", {}), arrays, "diag_blind")
        rows = capture.metric.get("rows") if vision else d.get("rows")
        weights = capture.metric.get("weights") if vision else d.get("weights")
        if rows is None:
            rows = d.get("rows", d.get("topk"))
        if weights is None:
            weights = d.get("weights")
        record.update(lib=d.get("lib"), rows=None if rows is None else np.asarray(rows).tolist(),
                      weights=None if weights is None else np.asarray(weights).tolist(),
                      scores=split_diagnostics(list(np.asarray(d.get("scores", [])).reshape(-1)), arrays),
                      conf=split_diagnostics(d.get("conf"), arrays), d1=ex.get("d1"),
                      k_eff=ex.get("k_eff"), member_k_eff=ex.get("w_eff"), unsupported_mass=ex.get("unsupported_mass"),
                      k_eff_status=schema.status("available" if "k_eff" in ex else "unsupported",
                                                "effective demo count needs the offline catalog"),
                      lib_sha=self.runtime.debug_library_shas.get(d.get("lib")),
                      retrieval_status=schema.status("available" if vision and rows is not None else "not_applicable",
                                                     "carried provenance; no live retrieval" if not vision else ""),
                      weights_status=schema.status("available" if weights is not None else "unsupported", "exact mixture weights"),
                      unsupported_mass_status=schema.status("available" if "unsupported_mass" in ex else "unsupported",
                                                           "not computed by serving method"))
        for src, dst in (("os_c_fresh", "eligible"), ("os_c_nominal_p", "p_nominal"), ("os_c_p", "p_effective"),
                         ("os_c_coin", "coin"), ("os_c_call", "treatment"), ("os_c_stall_state", "stall_state"),
                         ("os_c_cooldown", "cooldown"), ("os_c_rho", "budget_state")):
            if src in ex:
                record[dst] = ex[src]
        if "os_c_p" in ex:
            record["coin_domain"] = "dose-anchor"
            record["override"] = "cooldown" if ex.get("os_c_cooldown") else "confirmed_stall" if ex.get("os_c_stall_call") else None
        record["randomization_status"] = schema.status("available" if "coin" in record or "randomization" in d else "not_applicable",
                                                       "no recorded randomization" if "coin" not in record else "")
        if "randomization" in d:
            record["randomization"] = split_diagnostics(d["randomization"], arrays, "diag_randomization")
        try:
            diag, diag_status = debug_record(method)
            record["diag"] = split_diagnostics(diag, arrays)
            record["diag_status"] = diag_status
            # New methods use the shared protocol for assignment fields; keep
            # their full dictionary and also publish the schema's common names.
            for key in ("eligible", "p_nominal", "p_effective", "coin", "coin_domain", "treatment", "override",
                        "budget_state", "stall_state", "cooldown", "support", "propensities", "drawn_e",
                        "lottery_e", "fresh", "follow_source", "successor_rows", "random_seed"):
                if key in record["diag"]:
                    record[key] = record["diag"][key]
            if record["diag"].get("src") in ("cache", "policy", "policy_tail", "cache_tail", "follow"):
                record["src"] = record["diag"]["src"]
            if "coin" in record or "coin_domain" in record:
                record["randomization_status"] = schema.status("available", "recorded live assignment")
        except Exception as exc:
            record["diag"] = {}
            record["diag_status"] = schema.status("error", "{}: {}".format(type(exc).__name__, exc))
            capture.errors.append(record["diag_status"]["reason"])
        sampled = schema.rawkeys_sampled(self.campaign, record["task_uid"], record["decision_seq"])
        record.update(rawkeys_sampled=sampled,
                      snapshot_hint=schema.snapshot_sampled(self.campaign, record["task_uid"], record["decision_seq"]))
        if sampled and vision:
            for camera_name, field in (("third", "b_v0"), ("wrist", "b_v1")):
                if camera == "wrist_only" and camera_name == "third":
                    continue
                buf = getattr(session, field, None)
                if buf is not None and buf.n > step:
                    raw["raw_key_" + camera_name] = detached(buf.a[step], np.float32)
        record["rawkeys_status"] = schema.status("available" if raw else "not_sampled" if not sampled else "not_applicable",
                                                "hash sample" if not sampled else "no live raw keys" if not raw else "")
        record["array_status"] = {k: schema.status("available") for k in arrays}
        record["image_shapes"] = {camera_name: list(arrays["img_" + camera_name].shape)
                                  for camera_name in self.manifest["camera_names"] if "img_" + camera_name in arrays}
