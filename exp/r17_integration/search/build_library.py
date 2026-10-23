"""Complete-episode selection and dynamic stock CacheEntry artifact bridge."""
from __future__ import annotations

import os
from pathlib import Path
import pickle

import numpy as np

from exp.r17_integration.debug.reader_profiles import CaptureReader, read_h5
from exp.r17_integration.debug.validate import canonical, decision_identity, digest, episode_identity, file_sha
from .store import atomic_json, fsync_directory, write_store

RECIPES = {
    "teacher_library": "terminal teacher episodes; stored features/actions/native snapshots",
    "factor_warmup": "shared WarmupPool factor calibration; excluded from teacher/test episodes",
    "gpu_network_warmup": "model forward timing warmup; excluded from all episode outcomes",
}


def mw_split(orig_init_idx):
    idx = int(orig_init_idx)
    if 10 <= idx < 30:
        return "library"
    if 30 <= idx < 50:
        return "dev"
    if 0 <= idx < 10:
        return "test"
    raise ValueError("init outside sealed MW split")


def nested_episodes(episodes, sizes, *, task_names, success_policy="all"):
    """Deterministic quota per canonical task; never select using eval outcome."""
    if success_policy not in ("all", "success_only"):
        raise ValueError("explicit success selection policy required")
    tasks = tuple(sorted(task_names))
    if len(tasks) != len(set(tasks)) or not tasks:
        raise ValueError("task list must be canonical and unique")
    grouped = {task: [] for task in tasks}
    seen = set()
    for ep in episodes:
        key = episode_identity(ep)
        if key[:-1] in seen:
            raise ValueError("duplicate source episode/attempt")
        seen.add(key[:-1])
        if ep.get("split") != "library" or not ep.get("closed_ok") or not isinstance(ep.get("success"), bool):
            raise ValueError("library requires terminal library-only source episodes")
        if ep["task_name"] not in grouped:
            raise ValueError("unknown task")
        if success_policy == "all" or ep["success"]:
            grouped[ep["task_name"]].append(ep)
    for values in grouped.values():
        values.sort(key=lambda r: (r["orig_init_idx"], r["task_data_hash"], r["task_uid"]))
    result = {}
    for size in sorted(set(sizes)):
        if size <= 0 or size % len(tasks):
            raise ValueError("library size must be positive complete per-task quotas")
        quota = size // len(tasks)
        if any(len(v) < quota for v in grouped.values()):
            raise ValueError("insufficient complete source episodes")
        selected = [ep for task in tasks for ep in grouped[task][:quota]]
        result[size] = dict(episodes=selected, size=size, episodes_per_task=quota,
                            corrector_supported=quota >= 3, selection_policy=success_policy,
                            digest=digest(selected))
    return result


def build_library(sources, destination, binding, episodes, *, key_fields, collected_source_manifests=None,
                  feature_builder=None, split_policy=None):
    """Consume only closed captures and explicitly selected complete episodes."""
    if set(key_fields) != set(binding["model"]["feature_spec"]["fields"]):
        raise ValueError("key fields must exactly match real FeatureSpec fields")
    split_policy = split_policy or ("mt50_init_v1" if binding["benchmark_protocol"].startswith("mt50") else "declared_partition_v1")
    if split_policy not in ("mt50_init_v1", "declared_partition_v1"):
        raise ValueError("unknown sealed split policy")
    selected = {episode_identity(ep): ep for ep in episodes}
    if len(selected) != len(episodes):
        raise ValueError("duplicate selected episode")
    for ep in episodes:
        if ep.get("split") != "library" or not ep.get("closed_ok"):
            raise ValueError("fit source is not a closed library episode")
        if split_policy == "mt50_init_v1" and mw_split(ep["orig_init_idx"]) != "library":
            raise ValueError("MW library split leakage")
    captured, source_hashes = [], {}
    for source in map(Path, sources):
        if source.is_dir():
            reader = CaptureReader(source, expected_binding=binding)
            captured.extend(reader.decisions())
            source_hashes[str(source / "r17_capture.json")] = file_sha(source / "r17_capture.json")
        else:
            if collected_source_manifests is not None and str(source) in collected_source_manifests:
                from exp.r17_integration.debug.reader_profiles import read_collected_h5
                if feature_builder is None:
                    raise ValueError("stock collected H5 requires model-owned feature builder")
                _, rows = read_collected_h5(source, expected_binding=binding,
                                           source_manifest=collected_source_manifests[str(source)], feature_builder=feature_builder)
            else:
                _, rows = read_h5(source, expected_binding=binding)
            captured.extend(rows)
            source_hashes[str(source)] = file_sha(source)
    kept = [item for item in captured if episode_identity(item[0]) in selected]
    kept.sort(key=lambda item: decision_identity(item[0]))
    if len({decision_identity(item[0]) for item in kept}) != len(kept):
        raise ValueError("duplicate library decision")
    per_episode = {key: [] for key in selected}
    for record, _, _ in kept:
        per_episode[episode_identity(record)].append(int(record["decision_seq"]))
    for key, seqs in per_episode.items():
        expected = selected[key].get("n_decisions")
        if expected is None or seqs != list(range(int(expected))) or not seqs:
            raise ValueError("missing/noncontiguous complete episode decisions")
    rows, arrays = [], {}
    for record, values, keys in kept:
        payload = dict(values)
        for field in key_fields:
            if field not in keys:
                raise ValueError("missing real retrieval field: " + field)
            if keys[field].shape != (binding["model"]["feature_spec"]["fields"][field],):
                raise ValueError("retrieval geometry differs from FeatureSpec")
            payload["key_" + field] = keys[field]
        if "normalized_actions" not in payload or "state" not in payload:
            raise ValueError("library requires actions/state")
        sampler = binding.get("sampler", binding["model"]["default_sampler"])
        receipt = record.get("sampler_receipt", {})
        if receipt.get("identity") is not None and receipt["identity"] != digest(sampler):
            raise ValueError("actual capture sampler differs from library binding")
        if "native_grid" in payload and payload["native_grid"].tolist() != sampler["native_timesteps"]:
            raise ValueError("actual native grid differs from library binding")
        layout = binding["model"]["layout"]
        h = int(layout["horizon"])
        width = int(layout.get("model_action_width") or layout["valid_action_dim"])
        if payload["normalized_actions"].shape != (h, width):
            raise ValueError("normalized action geometry mismatch")
        if not np.isfinite(payload["normalized_actions"]).all() or not np.isfinite(payload["state"]).all():
            raise ValueError("nonfinite source state/action")
        active = layout.get("valid_action_indices") or list(range(layout["valid_action_dim"]))
        payload["active_actions"] = payload["normalized_actions"][:, active]
        indices = layout["valid_state_indices"]
        if payload["state"].ndim != 1 or max(indices) >= payload["state"].size:
            raise ValueError("source state/active layout mismatch")
        payload["active_state"] = payload["state"][indices]
        if rows and set(arrays) != set(payload):
            raise ValueError("missing source arrays; no implicit warm/RNG defaults")
        row = dict(record, split="library", success=selected[episode_identity(record)]["success"])
        rows.append(row)
        for name, value in payload.items():
            arrays.setdefault(name, []).append(value)
    links = {}
    for row in rows:
        row["row_id"] = digest(dict(binding=binding, decision=decision_identity(row)))
        row["episode_id"] = digest(episode_identity(row))
        links.setdefault(row["episode_id"], []).append(row)
    for group in links.values():
        for i, row in enumerate(group):
            row.update(prev_id=group[i-1]["row_id"] if i else None,
                       next_id=group[i+1]["row_id"] if i+1 < len(group) else None,
                       progress=i/max(1, len(group)-1))
    return write_store(destination, binding, rows, arrays, source_hashes=source_hashes,
                       metadata=dict(recipe="teacher_library", episodes=episodes, key_fields=list(key_fields), split_policy=split_policy))


def stock_artifact(store):
    """FULL_HIT stock artifact; native warm metadata stays in versioned store."""
    import torch
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CACHE_QUERY_FIELDS, CheckpointID
    fields = store.manifest["metadata"]["key_fields"]
    if not set(fields).issubset(CACHE_QUERY_FIELDS):
        raise ValueError("provider must map retrieval fields to stock names")
    entries, by_episode = [], {}
    vector_dims = {}
    for i, row in enumerate(store.rows):
        query_keys = {}
        for name in fields:
            v = np.asarray(store.array("key_" + name)[i])
            if v.ndim != 1 or not np.isfinite(v).all():
                raise ValueError("retrieval keys must be finite vectors")
            vector_dims[name] = v.size
            query_keys[name] = torch.from_numpy(v.astype(np.float32, copy=True)).contiguous()
        payload = CachePayload(torch.from_numpy(np.asarray(store.array("normalized_actions")[i]).astype(np.float32, copy=True)),
                               task_key=row["task_name"])
        identity = digest(dict(binding=store.manifest["binding"], decision=decision_identity(row)))
        trajectory = digest(episode_identity(row))
        entry = CacheEntry(identity, CheckpointID.CP1, query_keys, payload,
                           step_idx=row["decision_seq"], timestamp=0., trajectory_id=trajectory,
                           outcome=1 if row["success"] else -1)
        entries.append(entry)
        by_episode.setdefault(trajectory, []).append(entry)
    for group in by_episode.values():
        for i, entry in enumerate(group):
            entry.prev_ids = [group[i-1].id] if i else []
            entry.next_ids = [group[i+1].id] if i+1 < len(group) else []
    from exp.r17_integration.common.contracts import digest as shared_digest
    from exp.r17_integration.common.recipe import retrieval_model_identity
    b = store.manifest["binding"]
    s = b["model"]
    r17_binding = dict(model_id=s["model_id"], checkpoint_identity=s["checkpoint_identity"],
                       norm_identity=s["norm_identity"], layout_identity=shared_digest(s["layout"]),
                       feature_identity=shared_digest(s["feature_spec"]),
                       sampler_identity=shared_digest(b.get("sampler", s["default_sampler"])),
                       benchmark_protocol=b["benchmark_protocol"], retrieval_model_identity=retrieval_model_identity(s))
    return dict(version=1, key_builder_type="adapter_native", checkpoint_id="CP1",
                model=dict(model_spec=s, r17_binding=r17_binding), vector_dims=vector_dims, entries=entries,
                r17_manifest=store.manifest, native_warm_supported=False)


def save_stock_artifact(store, path):
    """Trusted locally-built pickle for the existing backend; never used for NPZ."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".part").open("xb") as f:
        pickle.dump(stock_artifact(store), f, protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path.with_name(path.name + ".part"), path)
    fsync_directory(path.parent)
    receipt = dict(sha256=file_sha(path), store_digest=store.manifest["content_digest"], binding=store.manifest["binding"])
    atomic_json(path.with_suffix(path.suffix + ".receipt.json"), receipt)
    return receipt


def load_stock_artifact(path, binding):
    """Verify our trusted pickle receipt before delegating to stock loader."""
    import json
    from exp.r17_integration.debug.validate import require_binding
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text())
    require_binding(receipt["binding"], binding)
    with path.open("rb") as f:
        artifact = pickle.load(f)
    require_binding(artifact["r17_manifest"]["binding"], binding)
    backend = InMemoryBackend(artifact["vector_dims"])
    backend.load_artifact(str(path))
    backend.artifact_meta.update(r17_manifest=artifact["r17_manifest"], native_warm_supported=False,
                                 native_sampler=binding.get("sampler", binding["model"]["default_sampler"]))
    # Stock's pre-schedule compatibility fallback is not a native sampler claim.
    backend.artifact_meta["schedule_id"] = None
    return backend
