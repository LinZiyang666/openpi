"""Default-off observe-only bridge to shared osdebug and H5 writers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

from .validate import canonical, decision_identity, digest, episode_identity, file_sha, numeric_arrays, plain


def image_array(value):
    """Recognize native uint8 HWC RGB without copying a tensor to the CPU."""
    return (getattr(value, "ndim", None) == 3 and value.shape[-1] == 3
            and str(getattr(value, "dtype", "")) in ("uint8", "torch.uint8"))


class CaptureBridge:
    def __init__(self, directory=None, *, enabled=False, binding=None, writer=None,
                 trace_sink=None, transport=None, queue_bytes=8 * 1024**2, capture_images=True):
        self.enabled = bool(enabled)
        self.capture_images = capture_images
        self.binding = plain(binding) if binding is not None else None
        self.directory = Path(directory) if directory is not None else None
        self.writer = writer
        self.trace_sink = trace_sink
        self.transport = transport
        self._lock = threading.Lock()
        self._closed = False
        self._wire_identities = {}
        if self.enabled and writer is None:
            if self.directory is None or binding is None:
                raise ValueError("enabled capture requires directory and model binding")
            from exp.offline_search.debug.server.writer import BlockWriter, atomic_json
            self.directory.mkdir(parents=True, exist_ok=True)
            manifest = self.directory / "r17_capture.json"
            if manifest.exists():
                raise FileExistsError("capture directory already initialized; use a new attempt directory")
            atomic_json(manifest, dict(schema="osdebug.v1", bridge="r17.p6.v1", binding=self.binding,
                                       status="partial", capability_statuses={}))
            self.writer = BlockWriter(self.directory, queue_bytes=queue_bytes)

    def capture(self, record, arrays, *, rawkeys=None, prompt="", trace=None):
        if not self.enabled:
            return None
        started = time.perf_counter()
        from exp.offline_search.debug.schema import decision_id, episode_key
        decision_identity(record)
        record = plain(record)
        record.setdefault("episode_key", episode_key(record["task_uid"], record["attempt"]))
        # Retain stock episode:dispatch:sequence wire format. The default
        # generation distinguishes provenance when a UID is reused across arms.
        record.setdefault("dispatch_gen", int(digest(episode_identity(record))[:12], 16))
        expected_id = decision_id(record["episode_key"], record["dispatch_gen"], record["decision_seq"])
        if record.get("decision_id", expected_id) != expected_id:
            raise ValueError("capture wire decision identity mismatch")
        record["decision_id"] = expected_id
        record.setdefault("capabilities", {})
        record.setdefault("capture_status", "available")
        serialization = dict(arrays={}, rawkeys={})
        if not self.capture_images:
            arrays = {k:v for k,v in arrays.items() if not image_array(v)}
            rawkeys = {k:v for k,v in (rawkeys or {}).items() if not image_array(v)}
        arrays = numeric_arrays(arrays, serialization=serialization["arrays"])
        rawkeys = numeric_arrays(rawkeys or {}, serialization=serialization["rawkeys"])
        if any(serialization.values()):
            record["array_serialization"] = serialization
        record.update(capture_enabled=True, pricing_eligible=False,
                      capture_bridge_prepare_ms=(time.perf_counter()-started)*1000,
                      capture_timing_scope="host_wall_bridge_prepare; native_capture/queue/trace excluded")
        with self._lock:
            if self._closed:
                raise RuntimeError("capture bridge closed")
            previous = self._wire_identities.get(expected_id)
            identity = decision_identity(record)
            if previous is not None:
                raise ValueError("duplicate/colliding capture wire decision identity")
            self._wire_identities[expected_id] = identity
            delay = self.writer.put(record, arrays, rawkeys, prompt=prompt)
            if trace is not None:
                if self.trace_sink is None:
                    raise ValueError("StepTrace supplied without shared H5TraceSink")
                self.trace_sink.record_step(trace)
        return delay

    def capture_physical(self, record, arrays):
        """P2 supplies native arrays and explicit semantics; preserve float64."""
        if not self.enabled:
            return None
        from exp.offline_search.debug.schema import write_npz_block
        if self.directory is None:
            raise ValueError("physical capture requires directory")
        decision_identity(record)
        idx = int(record["control_idx"])
        if idx < 0:
            raise ValueError("negative control index")
        serialization = {}
        arrays = numeric_arrays(arrays, serialization=serialization)
        if serialization:
            record = dict(record, array_serialization=dict(arrays=serialization))
        name = "physical_" + __import__("hashlib").sha256(canonical(record)).hexdigest() + ".npz"
        with self._lock:
            if self._closed:
                raise RuntimeError("capture bridge closed")
            path = self.directory / "physical" / name
            write_npz_block(path, arrays)
            row = dict(plain(record), block="physical/" + name, sha256=file_sha(path))
            with (self.directory / "physical.jsonl").open("ab") as f:
                f.write(canonical(row) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            if self.transport is not None:
                self.transport.publish(row, path)
        return row

    def close(self):
        if not self.enabled or self._closed:
            return None
        from exp.offline_search.debug.server.writer import atomic_json
        with self._lock:
            self.writer.close()
            self._closed = True
            stats = dict(self.writer.stats)
            if self.directory is not None:
                files = {str(p.relative_to(self.directory)): file_sha(p)
                         for p in self.directory.rglob("*") if p.is_file() and p.name != "r17_capture.json"}
                atomic_json(self.directory / "r17_capture.json",
                            dict(schema="osdebug.v1", bridge="r17.p6.v1", binding=self.binding,
                                 status="closed" if stats.get("drained") and not stats.get("errors") else "partial",
                                 writer_stats=stats, files=files))
            if self.transport is not None:
                self.transport.close()
        return stats


def export_h5(directory, destination):
    """Offline geometry-neutral H5 export; live H5 uses the shared TraceSink."""
    import h5py
    from .reader_profiles import CaptureReader
    reader = CaptureReader(directory)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    part = destination.with_name(destination.name + ".part")
    with h5py.File(part, "w") as f:
        f.attrs["r17_manifest"] = canonical(reader.manifest).decode()
        for i, (record, arrays, keys) in enumerate(reader.decisions()):
            group = f.create_group(f"step_{i:06d}")
            group.attrs["record"] = canonical(record).decode()
            for name, values in (("arrays", arrays), ("keys", keys)):
                g = group.create_group(name)
                for key, value in values.items():
                    if value.dtype.kind == "U":
                        g.create_dataset(key, data=value.astype(object), dtype=h5py.string_dtype("utf-8"))
                    else:
                        g.create_dataset(key, data=value)
        f.attrs["r17_closed_ok"] = reader.manifest["status"] == "closed"
        f.flush()
    with part.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(part, destination)
    return destination


class DecisionSink:
    """Consume P1 DecisionCapture with explicit episode provenance resolver."""
    def __init__(self, bridge, episode_metadata):
        self.bridge = bridge
        self.episode_metadata = episode_metadata
        self._pending = {}
        self._lock = threading.Lock()

    def __call__(self, capture):
        if not self.bridge.enabled:
            return
        request = capture.request
        metadata = self.episode_metadata(request) if callable(self.episode_metadata) else self.episode_metadata
        from .reader_profiles import canonical_episode, cost_fields
        metadata = canonical_episode(metadata)
        expected = dict(campaign=request.campaign, arm=request.arm, bundle_generation=request.bundle_generation,
                        benchmark_protocol=request.benchmark_protocol, task_name=request.task_name,
                        orig_init_idx=request.original_init_idx, attempt=request.attempt)
        if any(k in metadata and metadata[k] != value for k, value in expected.items()):
            raise ValueError("capture episode/request provenance identity disagreement")
        record = dict(metadata, campaign=request.campaign, arm=request.arm,
                      bundle_generation=request.bundle_generation, benchmark_protocol=request.benchmark_protocol,
                      task_name=request.task_name, orig_init_idx=request.original_init_idx,
                      attempt=request.attempt, decision_seq=request.decision_seq,
                      source=capture.source, cursor=capture.cursor, executed_count=None,
                      execution_status="pending_ack",
                      stage_calls=list(capture.stage_calls), actual_nfe=capture.actual_nfe,
                      sampler_receipt=plain(capture.sampler_receipt), diagnostics=plain(capture.diagnostics))
        record.update(cost_fields(record))
        arrays = dict(normalized_actions=capture.normalized_chunk, wire_actions=capture.wire_actions)
        if "robot_state" in capture.fields:
            arrays["state"] = capture.fields["robot_state"]
        sampler = self.bridge.binding["sampler"]
        grid = capture.sampler_receipt.get("native_timesteps")
        if grid is not None:
            arrays["native_grid"] = grid
        elif capture.source != "tail" and not capture.source.endswith("_tail"):
            # Do not claim an actual grid from default metadata when K differs.
            record["native_grid_status"] = "unsupported"
        record.setdefault("sampler_identity", sampler.get("sampler_id"))
        record["sampler_identity"] = capture.sampler_receipt.get("identity", record["sampler_identity"])
        record["native_snapshots"] = []
        def noise_metadata(value, path):
            from urllib.parse import quote
            if isinstance(value, dict):
                return {str(k): noise_metadata(v, path + "/" + str(k)) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [noise_metadata(v, path + "/" + str(i)) for i, v in enumerate(value)]
            if hasattr(value, "shape"):
                name = quote(path, safe="")
                arrays[name] = value
                return {"r17_array": name}
            return plain(value)
        for snapshot in getattr(capture, "native_snapshots", ()):
            from dataclasses import fields
            meta = {f.name: plain(getattr(snapshot, f.name)) for f in fields(snapshot)
                    if f.name not in ("x_t", "noise_state")}
            name = f"snapshot_{snapshot.loop_index}_x"
            arrays[name] = snapshot.x_t
            if snapshot.noise_state is not None:
                if isinstance(snapshot.noise_state, dict):
                    meta["noise_state"] = noise_metadata(snapshot.noise_state, f"snapshot/{snapshot.loop_index}/rng")
                else:
                    arrays[f"snapshot_{snapshot.loop_index}_rng"] = snapshot.noise_state
            meta["x_array"] = name
            record["native_snapshots"].append(meta)
        sample_capture = getattr(capture, "model_capture", {})
        record["native_capture_metadata"] = {}
        def flatten(value, prefix="native"):
            from urllib.parse import quote
            if isinstance(value, dict):
                for name, child in value.items():
                    flatten(child, prefix + "/" + str(name))
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
                for i, child in enumerate(value):
                    flatten(child, prefix + "/" + str(i))
            elif hasattr(value, "shape"):
                if self.bridge.capture_images or not image_array(value):
                    arrays[quote(prefix, safe="")] = value
            else:
                record["native_capture_metadata"][prefix] = plain(value)
        flatten(sample_capture)
        with self._lock:
            key = decision_identity(record)
            if key in self._pending:
                raise ValueError("duplicate pending capture decision identity")
            self._pending[key] = record
        self.bridge.capture(record, arrays, rawkeys=capture.fields)

    def on_execution(self, ack):
        if not self.bridge.enabled:
            return
        request = ack.get("request")
        with self._lock:
            matches = [key for key in self._pending if key[-1] == ack["decision_seq"]]
            if request is not None:
                request = plain(request)
                matches = [key for key in matches if (
                    key[0], key[1], key[2], key[3], key[4], key[5], key[8]) == (
                    request["campaign"], request["arm"], request["bundle_generation"], request["benchmark_protocol"],
                    request["task_name"], request["original_init_idx"], request["attempt"])]
            if len(matches) != 1:
                raise ValueError("ambiguous execution ack: associated RequestContext required")
            key = matches[0]
            row = self._pending[key]
        count = int(ack["executed_count"])
        cursor = int(ack["cursor"])
        if count < 0 or cursor != row["cursor"] + count:
            raise ValueError("execution ack count/cursor mismatch")
        with self._lock:
            self._pending.pop(key)
        row = dict(row, executed_count=count, cursor_end=cursor, execution_status="acknowledged",
                   terminal=bool(ack.get("terminal", False)))
        with self.bridge._lock:
            with (self.bridge.directory / "execution.jsonl").open("ab") as f:
                f.write(canonical(row) + b"\n")
                f.flush()
                os.fsync(f.fileno())


class PhysicalSink:
    """Adapt P2 event mappings; reset/settled/terminal stay metadata events."""
    def __init__(self, bridge, episode_metadata):
        self.bridge, self.episode_metadata = bridge, episode_metadata

    def __call__(self, event):
        if not self.bridge.enabled:
            return
        from exp.offline_search.debug.schema import write_npz_block
        metadata = self.episode_metadata(event) if callable(self.episode_metadata) else self.episode_metadata
        from .reader_profiles import canonical_episode
        metadata, event = canonical_episode(metadata), canonical_episode(event)
        from .validate import EPISODE_KEYS
        if any(k in metadata and k in event and metadata[k] != event[k] for k in EPISODE_KEYS):
            raise ValueError("physical episode/event provenance identity disagreement")
        record = dict(metadata, **event)
        episode_identity(record)
        physical = record.pop("physical")
        record.update(physical_metadata=physical.get("metadata", {}), capabilities=physical.get("capabilities", {}))
        for key in ("requested_action", "applied_action"):
            record.pop(key, None)  # P2 physical arrays preserve these separately.
        if record.get("event") == "control":
            self.bridge.capture_physical(record, physical["arrays"])
        else:
            # No fake decision_seq for reset or settle controls.
            from .validate import canonical, digest
            record["block"] = "physical/event_" + digest(record) + ".npz"
            with self.bridge._lock:
                if self.bridge._closed:
                    raise RuntimeError("capture bridge closed")
                block = self.bridge.directory / record["block"]
                serialization = {}
                arrays = numeric_arrays(physical["arrays"], serialization=serialization)
                if serialization:
                    record["array_serialization"] = dict(arrays=serialization)
                write_npz_block(block, arrays)
                record["sha256"] = file_sha(block)
                with (self.bridge.directory / "physical.jsonl").open("ab") as f:
                    f.write(canonical(record) + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
                if self.bridge.transport is not None:
                    self.bridge.transport.publish(record, block)


class SharedPhysicalTransport:
    """Use shared StreamSink's existing filename allowlist and spill protocol."""
    def __init__(self, stream_sink):
        self.sink = stream_sink
        self.sequence = 0

    def publish(self, record, block):
        # Number independent blocks; metadata supplies control_idx and decision_seq.
        name = f"controls_{self.sequence:04d}.npz"
        if self.sequence >= 10000:
            raise ValueError("shared control-block filename capacity exceeded")
        self.sequence += 1
        self.sink.write_file(name, Path(block).read_bytes())
        self.sink.write((canonical(dict(record, block=name)).decode()) + "\n")

    def close(self):
        self.sink.close()


def capture_native_product(bridge, product, episode_metadata):
    """Accept named-array products from model trace adapters without model branches.

    Hierarchical numeric names are reversibly encoded for flat store filenames.
    The caller supplies episode provenance; diagnostic probes cannot acquire a
    teacher episode identity implicitly.
    """
    if not bridge.enabled:
        return
    from urllib.parse import quote
    import numpy as np
    from .validate import model_binding, require_binding
    metadata = product.get("metadata", {})
    spec = metadata.get("model_spec", product.get("model_spec"))
    sampler = metadata.get("sampler", product.get("native_grid"))
    if spec is not None:
        require_binding(model_binding(spec, bridge.binding["benchmark_protocol"], sampler=sampler), bridge.binding)
    record = dict(episode_metadata, native_product_metadata=plain(metadata))
    record["actual_nfe"] = metadata.get("actual_nfe", product.get("actual_nfe", record.get("actual_nfe")))
    values, keys, paths = {}, {}, {}
    def flatten(value, path):
        if isinstance(value, dict):
            for name, child in value.items():
                flatten(child, path + "/" + str(name) if path else str(name))
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            for i, child in enumerate(value):
                flatten(child, path + "/" + str(i))
        elif hasattr(value, "shape"):
            flat = quote(path, safe="")
            values[flat] = value
            paths[flat] = path
        else:
            record.setdefault("native_scalar_metadata", {})[path] = plain(value)
    for name, value in product.get("arrays", {}).items():
        if name.startswith("keys/"):
            keys[name.split("/", 1)[1]] = value
        elif name.startswith("key_"):
            keys[name[4:]] = value
        else:
            flatten(value, name)
    keys.update(product.get("retrieval_fields", product.get("fields", {})))
    for name in ("normalized_actions", "state", "wire_actions", "tokens", "cp2_fields", "native_snapshots", "native_steps"):
        if name in product:
            flatten(product[name], name)
    if "robot_state" in product:
        values.setdefault("state", product["robot_state"])
    if "robot_state" in keys:
        values.setdefault("state", keys["robot_state"])
    if isinstance(sampler, dict) and "native_timesteps" in sampler:
        values.setdefault("native_grid", np.asarray(sampler["native_timesteps"]))
        if sampler.get("native_deltas"):
            values["native_deltas"] = np.asarray(sampler["native_deltas"])
    record["array_paths"] = paths
    bridge.capture(record, values, rawkeys=keys)


class CaptureFactory:
    """Connection-local sinks over binding-shared writers, using P1's factory.

    This shim avoids ambiguous seq0 acknowledgments even before P1 appends a
    RequestContext to its execution callback. No model wrapper is patched.
    """
    def __init__(self, provider, config, storage, *, directory=None, episode_metadata=None, enabled=False, capture_images=True, **kwargs):
        from exp.r17_integration.common.serving import make_connection_factory
        self.provider, self.config, self.storage = provider, config, storage
        self.directory, self.episode_metadata = directory, episode_metadata
        self.enabled, self.kwargs = enabled, kwargs
        self.capture_images = capture_images
        self.base_factory = make_connection_factory(provider, config, storage, **kwargs)
        self.bridges = {}
        self._lock = threading.Lock()

    def __call__(self, base_policy=None, bundle_id="default"):
        if not self.enabled:
            return self.base_factory(base_policy, bundle_id)
        from exp.r17_integration.common.serving import make_bundle_validator
        from exp.r17_integration.common.provider import validate_binding
        from openpi.serving.websocket_policy_server import get_current_cache_bundle
        from .validate import digest, model_binding
        config, storage = self.config, self.storage
        if bundle_id != "default":
            bundle = get_current_cache_bundle(bundle_id)
            if bundle is None:
                raise ValueError("unknown capture bundle")
            config, storage = bundle.cache_config, bundle.shared_storage
        make_bundle_validator(self.provider)(config, storage)
        _, sampler = validate_binding(config.adapter_binding)
        binding = model_binding(self.provider.model_spec, config.adapter_binding["benchmark_protocol"], sampler=sampler)
        if self.directory is None:
            raise ValueError("enabled CaptureFactory needs output root")
        key = digest(binding)
        with self._lock:
            if key not in self.bridges:
                self.bridges[key] = CaptureBridge(Path(self.directory) / ("binding_" + key[:16]), enabled=True, binding=binding,
                                                  capture_images=self.capture_images)
            bridge = self.bridges[key]
        core = self.base_factory(base_policy, bundle_id)
        sink = DecisionSink(bridge, self.episode_metadata or (lambda request: dict(core._episode)))
        from exp.r17_integration.common.contracts import CaptureSpec
        core.spec.capabilities.require("debug_capture")
        core.capture_sink = sink
        core.capture = CaptureSpec(enabled=True, snapshots=True)
        return core

    def close(self):
        with self._lock:
            return [bridge.close() for bridge in self.bridges.values()]


class CoreCaptureFactory:
    """Attach to an already built connection core; never replace its recipe."""
    def __init__(self, directory, *, provider=None, enabled=True, capture_images=True):
        self.directory, self.provider, self.enabled = directory, provider, enabled
        self.capture_images = capture_images
        self.bridges, self._lock = {}, threading.Lock()

    def __call__(self, core):
        if not self.enabled:
            return None
        from .validate import model_binding
        if self.provider is not None and core.spec.identity != self.provider.model_spec.identity:
            raise ValueError("capture core belongs to another model provider")
        binding = model_binding(core.spec, core.config.adapter_binding["benchmark_protocol"], sampler=core.sampler)
        key = digest(binding)
        with self._lock:
            if key not in self.bridges:
                self.bridges[key] = CaptureBridge(Path(self.directory) / ("binding_"+key[:16]), enabled=True, binding=binding,
                                                  capture_images=self.capture_images)
            bridge = self.bridges[key]
        return DecisionSink(bridge, lambda request: dict(core._episode))

    def close(self):
        with self._lock:
            return [bridge.close() for bridge in self.bridges.values()]


def make_capture_sink_factory(provider, directory=None, *, enabled=True, capture_images=True):
    """Provider+directory form or directory-only compatibility with P2's shim."""
    if directory is None:
        directory, provider = provider, None
    return CoreCaptureFactory(directory, provider=provider, enabled=enabled, capture_images=capture_images)


class PhysicalCaptureFactory:
    """Client-only worker capture selected from attested model and task sampler."""
    def __init__(self, directory, server_metadata, *, sampler_resolver=None, enabled=True):
        self.directory, self.enabled, self.sampler_resolver = directory, enabled, sampler_resolver
        self.spec = plain(server_metadata.get("r17_model_spec"))
        if enabled and not self.spec:
            raise ValueError("worker capture requires server r17_model_spec")
        self.bridges, self._lock = {}, threading.Lock()

    def __call__(self, event):
        if not self.enabled:
            return
        from .validate import model_binding
        sampler = event.get("sampler")
        if sampler is None and "num_steps" in event:
            if self.sampler_resolver:
                sampler = self.sampler_resolver(event["num_steps"])
            elif event["num_steps"] == self.spec["default_sampler"]["num_steps"]:
                sampler = self.spec["default_sampler"]
        if sampler is None:
            raise ValueError("worker capture requires explicit actual task sampler")
        sampler = plain(sampler)
        if (any(k not in sampler for k in ("sampler_id", "family", "num_steps", "native_timesteps"))
                or type(sampler["num_steps"]) is not int or sampler["num_steps"] <= 0
                or len(sampler["native_timesteps"]) != sampler["num_steps"]):
            raise ValueError("incomplete actual task sampler")
        if "num_steps" in event and event["num_steps"] != sampler["num_steps"]:
            raise ValueError("worker sampler step count mismatch")
        if event.get("sampler_identity") not in (None, sampler["sampler_id"], digest(sampler)):
            raise ValueError("worker sampler identity mismatch")
        protocol = event["benchmark_protocol"]
        if protocol not in self.spec.get("allowed_benchmark_protocols", ()):
            raise ValueError("worker capture benchmark protocol mismatch")
        binding = model_binding(self.spec, protocol, sampler=sampler)
        key = digest(binding)
        with self._lock:
            if key not in self.bridges:
                self.bridges[key] = CaptureBridge(Path(self.directory) / ("binding_"+key[:16]), enabled=True, binding=binding)
            bridge = self.bridges[key]
        PhysicalSink(bridge, {})(event)

    def close(self):
        with self._lock:
            return [bridge.close() for bridge in self.bridges.values()]
