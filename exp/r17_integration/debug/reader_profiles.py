"""Lossless generic reader; legacy readers remain available explicitly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .validate import EPISODE_KEYS, decision_identity, episode_identity, plain, require_binding


def canonical_episode(row):
    """Canonicalize published runner aliases; contradictory aliases fail closed."""
    row = dict(row)
    for target, aliases in {"orig_init_idx": ("orig_init_state_idx", "original_init_idx"),
                            "task_name": ("task",)}.items():
        for alias in aliases:
            if alias in row:
                if target in row and row[target] != row[alias]:
                    raise ValueError("episode identity aliases disagree: " + target)
                row[target] = row[alias]
    return row


def cost_fields(row):
    """Consume P1 diagnostics/meta or P2 server wall clock without guessing zero."""
    fields = {}
    sources = (row.get("diagnostics", {}), row.get("__hit_meta__", {}), row)
    for source in sources:
        for key in ("source", "cursor", "stage_calls", "actual_nfe",
                    "capture_enabled", "pricing_eligible", "capture_overhead_ms", "capture_overhead_scope", "capture_sink_ms", "capture_bridge_prepare_ms",
                    "capture_timing_scope", "sampler_identity", "commit_horizon"):
            if source.get(key) is not None:
                fields[key] = plain(source[key])
    # One decision's timing must come from ONE ledger: the policy's own
    # diagnostics/meta win over top-level copies. In closed osdebug captures the
    # top-level ``queue_wait_ms`` is the debug WRITER's backpressure wait
    # (BlockWriter.put), and episode metadata may carry unrelated copies.
    for key in ("cost_ms", "stage_ms", "stage_timing_ms", "timing_scope", "timing_kind",
                "queue_wait_ms", "service_ms", "stage_queue_wait_ms", "stage_service_ms", "queue_wait_status",
                "pricing_time_field", "service_timing_scope", "gpu_service_ms"):
        for source in sources:
            if source.get(key) is not None:
                fields[key] = plain(source[key])
                break
    policy_ledger = any(row.get(name) for name in ("diagnostics", "__hit_meta__"))
    if policy_ledger and row.get("queue_wait_ms") is not None and fields.get("queue_wait_ms") != row["queue_wait_ms"]:
        fields["writer_queue_wait_ms"] = plain(row["queue_wait_ms"])
    if "stage_timing_ms" in fields:
        fields.setdefault("stage_ms", fields["stage_timing_ms"])
    if "timing_kind" in fields:
        fields.setdefault("timing_scope", fields["timing_kind"])
    timing = row.get("server_timing", {})
    if timing.get("infer_ms") is not None:
        fields["server_infer_ms"] = timing["infer_ms"]
        if fields.get("cost_ms") is None:
            fields.update(cost_ms=timing["infer_ms"], timing_scope="server_infer_host_wall")
    if fields.get("capture_enabled") is True:
        fields["pricing_eligible"] = False
    if fields.get("pricing_time_field") not in (None, "service_ms"):
        raise ValueError("unknown pricing_time_field: " + str(fields["pricing_time_field"]))
    if fields.get("pricing_time_field") == "service_ms" and fields.get("service_ms") is None:
        raise ValueError("pricing_time_field names service_ms but no service time was measured")
    if fields.get("service_ms") is None or fields.get("queue_wait_ms") is None:
        # Host latency without a measured queue/service split is diagnostic only.
        fields["pricing_eligible"] = False
    for key in ("cost_ms", "capture_overhead_ms", "capture_sink_ms", "capture_bridge_prepare_ms", "server_infer_ms",
                "queue_wait_ms", "service_ms"):
        if key in fields and (not isinstance(fields[key], (int, float)) or isinstance(fields[key], bool)
                              or not np.isfinite(fields[key]) or fields[key] < 0):
            raise ValueError("invalid measured cost/timing field: " + key)
    if "stage_calls" in fields and (len(fields["stage_calls"]) != 3 or
            any(type(v) is not int or v < 0 for v in fields["stage_calls"])):
        raise ValueError("invalid stage call cost ledger")
    for name in ("actual_nfe", "cursor"):
        if name in fields and (type(fields[name]) is not int or fields[name] < 0):
            raise ValueError("invalid cost ledger: " + name)
    for name in ("stage_ms", "stage_queue_wait_ms", "stage_service_ms"):
        if fields.get(name) is not None:
            values = list(fields[name].values()) if isinstance(fields[name], dict) else fields[name]
            # Service = wall - wait can be a few ulp negative; anything else is a broken ledger.
            if not all(isinstance(v, (int, float)) and np.isfinite(v) and v >= -1e-6 for v in values):
                raise ValueError("invalid measured stage timing: " + name)
    if (fields.get("service_ms") is not None and fields.get("queue_wait_ms") is not None
            and fields.get("cost_ms") is not None
            and abs(fields["service_ms"] + fields["queue_wait_ms"] - fields["cost_ms"]) > 1e-3):
        raise ValueError("service_ms + queue_wait_ms must equal host wall cost_ms")
    return fields


def merge_decision_ledgers(decisions, *, cost_rows=(), execution_rows=(), episode_manifests=()):
    """Exact server/client decision join, including the final client execution ack.

    P2 per_step_rows may be supplied as cost_rows: execution_ack entries are
    separated, episode_summary is ignored. Sparse stock rows need a sealed
    episode manifest keyed by task_uid/attempt; never join on sequence alone.
    """
    manifests = {}
    for manifest in episode_manifests:
        manifest = canonical_episode(manifest)
        key = manifest["task_uid"], manifest["attempt"]
        episode_identity(manifest)
        if key in manifests:
            raise ValueError("duplicate episode manifest identity")
        manifests[key] = manifest
    def enrich(row):
        row = canonical_episode(row)
        manifest = manifests.get((row.get("task_uid"), row.get("attempt")))
        if manifest is not None:
            if any(row.get(k, manifest[k]) != manifest[k] for k in EPISODE_KEYS):
                raise ValueError("ledger/manifest episode identity disagreement")
            row = dict(manifest, **row)
        decision_identity(row)
        return row
    result, index = [], {}
    for row in decisions:
        row = enrich(row)
        row.update(cost_fields(row))
        key = decision_identity(row)
        if key in index:
            raise ValueError("duplicate decision identity")
        index[key] = row
        result.append(row)
    execution_rows = list(execution_rows)
    seen_cost = set()
    for row in cost_rows:
        if row.get("_kind") == "episode_summary":
            continue
        if row.get("_kind") == "execution_ack":
            execution_rows.append(row)
            continue
        row = enrich(row)
        key = decision_identity(row)
        if key not in index:
            raise ValueError("unmatched cost ledger identity")
        if key in seen_cost:
            raise ValueError("duplicate cost ledger identity")
        seen_cost.add(key)
        target = index[key]
        update = cost_fields(row)
        for name in ("stage_calls", "actual_nfe", "source", "cursor", "sampler_identity"):
            if target.get(name) is not None and name in update and plain(target[name]) != plain(update[name]):
                raise ValueError("cost ledger decision disagreement: " + name)
        contaminated = target.get("capture_enabled") is True or update.get("capture_enabled") is True
        target.update(update)
        if contaminated:
            target.update(capture_enabled=True, pricing_eligible=False)
    seen_ack = set()
    for ack in execution_rows:
        ack = enrich(ack)
        key = decision_identity(ack)
        if key not in index:
            raise ValueError("unmatched execution ledger identity")
        if key in seen_ack:
            raise ValueError("duplicate execution ledger identity")
        seen_ack.add(key)
        target = index[key]
        count = ack.get("executed_count")
        if type(count) is not int or count < 0 or target.get("cursor") is None:
            raise ValueError("invalid execution count/cursor")
        end = target["cursor"] + count
        declared = ack.get("cursor_end", ack.get("cursor", end))
        if "last_cursor" in ack and ack["last_cursor"] != end - 1 or declared != end:
            raise ValueError("execution count/cursor mismatch")
        if target.get("executed_count") is not None and target["executed_count"] != count:
            raise ValueError("server/client execution disagreement")
        if target.get("cursor_end") is not None and target["cursor_end"] != end:
            raise ValueError("server/client execution cursor disagreement")
        target.update(executed_count=count, cursor_end=end, execution_status="acknowledged",
                      terminal=bool(ack.get("terminal", False)))
    for row in result:
        count = row.get("executed_count")
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("invalid execution count")
        if count is not None and row.get("cursor_end") is not None and row.get("cursor") is not None:
            if row["cursor_end"] != row["cursor"] + count:
                raise ValueError("execution count/cursor mismatch")
    return result


def accepted_episodes(journal):
    accepted = {}
    for row in journal:
        if (row.get("accepted") is not True or row.get("error") is not None or
                row.get("status") not in ("ok", "completed", "done", "failed")):
            continue
        row = canonical_episode(row)
        key = episode_identity(row)
        uid = key[:-1]
        if uid in accepted:
            raise ValueError("multiple accepted attempts for episode")
        if not isinstance(row.get("success"), bool):
            raise ValueError("accepted result missing success")
        accepted[uid] = row
    return accepted


def enrich_journal(journal, episode_manifests):
    """Stock journals retain their original schema; sealed manifests add identity."""
    lookup = {}
    for manifest in episode_manifests:
        manifest = canonical_episode(manifest)
        episode_identity(manifest)
        key = manifest["task_uid"], manifest["attempt"]
        if key in lookup:
            raise ValueError("duplicate episode manifest")
        lookup[key] = manifest
    result = []
    for row in journal:
        row = canonical_episode(row)
        key = row["task_uid"], row.get("attempt")
        if key not in lookup:
            if row.get("accepted") is True:
                raise ValueError("accepted journal lacks exact attempt manifest")
            result.append(row)
            continue
        manifest = lookup[key]
        for name in manifest.keys() & row.keys():
            if name in ("campaign", "arm", "bundle_generation", "benchmark_protocol", "task_name",
                        "orig_init_idx", "task_data_hash", "task_uid", "attempt") and manifest[name] != row[name]:
                raise ValueError("journal/manifest identity disagreement")
        result.append(dict(manifest, **row))
    return result


def accepted_join(decisions, journal):
    accepted = accepted_episodes(journal)
    result, seen = [], set()
    for row in decisions:
        identity = decision_identity(row)
        episode = accepted.get(identity[:-2])
        if episode is None:
            if any(r["task_uid"] == row["task_uid"] and r["attempt"] == row["attempt"] for r in accepted.values()):
                raise ValueError("captured/accepted episode provenance identity disagreement")
            continue
        if episode["attempt"] != row["attempt"]:
            continue
        if identity in seen:
            raise ValueError("duplicate accepted decision")
        seen.add(identity)
        result.append(dict(row, success=episode["success"]))
    return result


class CaptureReader:
    def __init__(self, directory, *, expected_binding=None, allow_partial=False, max_cached_blocks=None):
        if max_cached_blocks is not None and (type(max_cached_blocks) is not int or max_cached_blocks < 2):
            raise ValueError('capture block cache must hold at least2 blocks')
        self.max_cached_blocks = max_cached_blocks
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "r17_capture.json").read_text())
        self.issues = []
        if self.manifest["bridge"] != "r17.p6.v1":
            raise ValueError("unknown capture bridge version")
        if expected_binding is not None:
            require_binding(self.manifest["binding"], expected_binding)
        if self.manifest["status"] != "closed" and not allow_partial:
            raise ValueError("partial capture cannot become a library")
        for name in self.manifest.get("files", {}):
            path = self.directory / name
            if not path.resolve().is_relative_to(self.directory.resolve()):
                raise ValueError("capture path escapes directory: " + name)

    def decisions(self):
        from exp.offline_search.debug.schema import read_jsonl, read_npz_block
        blocks = {}
        seen = set()
        acknowledgments = {}
        ack_path = self.directory / "execution.jsonl"
        if ack_path.exists():
            for ack in read_jsonl(ack_path, self.issues):
                identity = decision_identity(ack)
                if identity in acknowledgments:
                    raise ValueError("duplicate execution ack")
                acknowledgments[identity] = ack
        for path in sorted(self.directory.glob("decisions_*.jsonl")):
            for row in read_jsonl(path, self.issues):
                identity = decision_identity(row)
                if identity in seen:
                    raise ValueError("duplicate captured decision")
                seen.add(identity)
                if identity in acknowledgments:
                    ack = acknowledgments[identity]
                    row.update({key: ack[key] for key in ("executed_count", "cursor_end", "execution_status", "terminal")})
                if row.get("status") == "error":
                    self.issues.append(dict(reason="writer error", record=row))
                    continue
                def extract(folder, field, index):
                    if field not in row:
                        return {}
                    block = self.directory / folder / row[field]
                    if block.parent.resolve() != (self.directory / folder).resolve():
                        raise ValueError("block path escapes capture directory")
                    if block not in blocks:
                        if self.max_cached_blocks is not None and len(blocks) >= self.max_cached_blocks:
                            blocks.pop(next(iter(blocks)))
                        blocks[block] = read_npz_block(block)
                    values = blocks[block]
                    i = int(row[index])
                    if str(values["decision_id"][i]) != row["decision_id"]:
                        raise ValueError("block identity mismatch")
                    return {k: v[i].copy() for k, v in values.items()
                            if k not in ("decision_id", "prompts", "prompt_idx") and not k.endswith("_available")
                            and bool(values.get(k + "_available", np.ones(len(values["decision_id"]), bool))[i])}
                yield row, extract("blocks", "blk", "blk_i"), extract("rawkeys", "rawkeys_blk", "rawkeys_blk_i")

    def physical(self):
        from exp.offline_search.debug.schema import read_jsonl, read_npz_block
        path = self.directory / "physical.jsonl"
        if path.exists():
            for row in read_jsonl(path, self.issues):
                block = self.directory / row["block"]
                if not block.resolve().is_relative_to(self.directory.resolve()):
                    raise ValueError("physical block path escapes directory")
                yield row, read_npz_block(block)


def read_h5(path, *, expected_binding=None):
    import h5py
    with h5py.File(path, "r") as f:
        manifest = json.loads(f.attrs["r17_manifest"])
        if not bool(f.attrs.get("r17_closed_ok", False)):
            raise ValueError("partial H5 cannot become a library")
        if expected_binding is not None:
            require_binding(manifest["binding"], expected_binding)
        rows = []
        for name in sorted(f):
            g = f[name]
            def values(kind):
                return {k: (v.asstr()[()] if h5py.check_string_dtype(v.dtype) else v[()])
                        for k, v in g[kind].items()}
            rows.append((json.loads(g.attrs["record"]), values("arrays"), values("keys")))
    return manifest, rows


def open_legacy_arm(run_root, arm_name):
    from exp.offline_search.debug.reader import open_arm
    return open_arm(run_root, arm_name)


def read_collected_h5(path, *, expected_binding, source_manifest, feature_builder):
    """Read stock DataCollector/TraceSink H5 using model-owned key extraction.

    Old H5 has no complete model binding. Require an external sealed source
    manifest instead of treating its camera names as a new-model certificate.
    """
    import h5py
    from urllib.parse import quote
    require_binding(source_manifest["binding"], expected_binding)
    if not source_manifest["episode"].get("closed_ok"):
        raise ValueError("collected H5 source lacks closed content attestation")
    episode = source_manifest["episode"]
    rows = []
    with h5py.File(path, "r") as f:
        if "trace_schema_version" in f.attrs and not (
                f.attrs.get("trace_closed_ok", False) and f.attrs.get("trace_terminal", False)):
            raise ValueError("partial shared trace cannot become a library")
        names = sorted(name for name in f if name.startswith("step_"))
        if len(names) != int(episode["n_decisions"]) or int(f.attrs["num_steps"]) != len(names):
            raise ValueError("collected H5 step count mismatch")
        for seq, name in enumerate(names):
            group = f[name]
            raw = {}
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "biufc":
                    raw[name] = obj[()]
            group.visititems(collect)
            keys = feature_builder(raw)
            arrays = {quote(name, safe=""): value for name, value in raw.items()}
            arrays["state"] = raw["robot_state"]
            arrays["normalized_actions"] = raw["clean_action"]
            record = dict(episode, decision_seq=seq, source="teacher", native_grid_status="unsupported")
            if "native_grid" in raw:
                record["native_grid_status"] = "available"
            rows.append((record, arrays, keys))
    return dict(binding=expected_binding, status="closed"), rows
