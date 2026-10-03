"""Resumable deferred augmentation; publication is confined to each arm's aug/."""
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .. import reader, schema
from .models import FakeModel, Pi05Model, GrootModel, Profile, input_sha, measure
from .retrieval import FrozenRetrieval, FakeRetrieval

KINDS = ("policy_shadow", "policy_draws", "shadow_look", "camera_shadow")


def private_seed(campaign, task_id, init, decision_seq, draw):
    payload = "%s|%d|%d|%d|%d" % (campaign, int(task_id), int(init), int(decision_seq), int(draw))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def code_sha():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("job.py", "models.py", "retrieval.py"):
        digest.update(name.encode() + (root / name).read_bytes())
    return digest.hexdigest()


def fit_specs_sha(retrieval, kind):
    if kind not in ("shadow_look", "camera_shadow"):
        return "none"
    specs = retrieval.fit_specs(kind) if hasattr(retrieval, "fit_specs") else dict(libraries=retrieval.libraries, fits=retrieval.provenance)
    return hashlib.sha256(json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def wire_observations(arm, records):
    ids = [r["decision_id"] for r in records]
    camera_map = arm.server_meta.get("camera_wire_keys", {"third": "observation/image", "wrist": "observation/wrist_image"})
    keys = ["img_" + camera for camera in camera_map] + ["state_wire", "prompt"]
    data = arm.decision_arrays(keys, ids)
    observations = []
    for i in range(len(ids)):
        obs = {wire: data["img_" + camera][i].copy() for camera, wire in camera_map.items()}
        obs["observation/state"] = data["state_wire"][i].copy()
        obs["prompt"] = str(data["prompt"][i])
        observations.append(obs)
    return observations


def _done(aug_dir, kind, provenance, upgrades=None):
    ids = set()
    for path in sorted((aug_dir / kind).glob("part_*.npz")):
        required = {"policy_shadow": ("chunk", "seed", "input_sha"), "policy_draws": ("chunks", "seeds"),
                    "shadow_look": ("rows", "weights", "scores", "cache_chunk", "lib"),
                    "camera_shadow": tuple(mode + "_" + k for mode in ("wrist", "third") for k in ("rows", "weights", "scores", "cache_chunk"))}[kind]
        data = reader.read_npz(path, ["_meta_json", "decision_id"] + list(required))
        metadata = json.loads(str(data["_meta_json"]))
        for key in ("model", "checkpoint_sha", "code_sha", "fit_config_sha", "dtype"):
            if metadata.get(key) != provenance.get(key):
                if key == "code_sha" and upgrades is not None and metadata.get(key):
                    upgrades.append(dict(path=str(path.relative_to(aug_dir)), old_code_sha=metadata[key],
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         decisions=len(data["decision_id"]), kind=kind))
                    continue
                raise ValueError("resumption provenance mismatch %s in %s" % (key, path))
        if "decision_id" not in data:
            raise ValueError("missing decision_id in " + str(path))
        new_ids = list(data["decision_id"].astype(str))
        if len(set(new_ids)) != len(new_ids) or ids.intersection(new_ids):
            raise ValueError("duplicate published augmentation decision_id: " + str(path))
        for field in required:
            if field not in data or not data[field].ndim or len(data[field]) != len(new_ids):
                raise ValueError("incomplete published %s field %s in %s" % (kind, field, path))
        ids.update(new_ids)
    return ids


def _atomic_json(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with part.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(part), str(target))


def _record_upgrade(aug_dir, upgrades, provenances):
    if not upgrades:
        return None
    note = dict(authorization="explicit --allow-code-upgrade", new_code_sha=code_sha(),
                kept_parts=upgrades, provenance_by_kind=provenances,
                explanation="Finished parts retain their original metadata and bytes. New parts use the new code SHA; checkpoint and used-fit identities remain strict.")
    fingerprint = hashlib.sha256(json.dumps(note, sort_keys=True).encode()).hexdigest()
    relative = "provenance/code_upgrade_%s.json" % fingerprint
    target = aug_dir / relative
    if not target.exists():
        _atomic_json(target, dict(note, recorded_unix_seconds=time.time()))
    return relative


def _publish(aug_dir, kind, arrays, provenance, batch_size):
    directory = aug_dir / kind
    directory.mkdir(parents=True, exist_ok=True)
    indices = [int(p.stem.split("_")[1]) for p in directory.glob("part_*.npz")]
    target = directory / ("part_%05d.npz" % (max(indices, default=-1) + 1))
    if target.exists():
        raise FileExistsError(str(target))
    metadata = dict(provenance, batch_size=batch_size, n_decisions=len(arrays["decision_id"]))
    arrays = dict(arrays, _meta_json=np.array(json.dumps(metadata, sort_keys=True)))
    schema.write_npz_block(target, arrays)
    return target


def _select(encoded, positions):
    """Select CPU key tensors only; model draw uses the complete stage batch."""
    return {k: encoded[k][positions] for k in ("v0", "v1", "state")}


def _history(records, encoded, served, observations, history):
    result = {}
    for i, record in enumerate(records):
        result[record["decision_id"]] = dict(
            v0=np.array(history["v0"][-1:], np.float32).reshape(-1, encoded["v0"].shape[1]),
            v1=np.array(history["v1"][-1:], np.float32).reshape(-1, encoded["v1"].shape[1]),
            state=np.array(history["state"], np.float32).reshape(-1, encoded["state"].shape[1]),
            served_chunk=np.asarray(history["actions"], np.float32),
            prev_hit=history.get("prev_hit"), raw_state=observations[i]["observation/state"])
        mode = record.get("camera_mode", "full")
        consumed = record.get("vision", False)
        for name, raw_key, excluded in (("v0", "v0", "wrist_only"), ("v1", "v1", "third_only")):
            value = encoded[raw_key][i] if consumed and mode != excluded else (history[name][-1] if history[name] else encoded[raw_key][i])
            history[name].append(value)
        history["state"].append(encoded["state"][i])
        history["actions"].append(served[i])
        history["prev_hit"] = bool(record.get("hit", True))
    return result


def run_arm(arm, model, retrieval=None, batch_size=8, kinds=None, checkpoint_sha=None, fit_config_sha="none", limit=None,
            allow_code_upgrade=False, prefetch=True, profile=False, output_dir=None):
    """One augmentation publisher per arm; reads never create derived caches."""
    aug_dir = Path(output_dir) if output_dir is not None else arm.debug_dir / "aug"
    aug_dir.mkdir(parents=True, exist_ok=True)
    with (aug_dir / ".job.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another augmentation job owns " + str(aug_dir)) from exc
        previous = arm.cache_enabled
        arm.cache_enabled = False
        try:
            return _run_arm(arm, model, retrieval, batch_size, kinds, checkpoint_sha, fit_config_sha, limit,
                            aug_dir, allow_code_upgrade, prefetch, profile)
        finally:
            arm.cache_enabled = previous


def _run_arm(arm, model, retrieval=None, batch_size=8, kinds=None, checkpoint_sha=None, fit_config_sha="none", limit=None,
             aug_dir=None, allow_code_upgrade=False, prefetch=True, profile=False):
    """Process an arm with an already loaded model (reused across priority arms).

    Existing published IDs are skipped. A partial episode's prefix may be
    encoded again solely to reconstruct factual retrieval history. Policy
    outputs are never recomputed for completed IDs. ``limit`` is an explicit
    smoke limit, never used by full completeness validation.
    """
    if batch_size < 1 or not checkpoint_sha:
        raise ValueError("positive batch_size and checkpoint SHA required")
    kinds = list(kinds or (KINDS if model.model == "pi05" else KINDS[:-1]))
    if any(kind not in KINDS for kind in kinds):
        raise ValueError("unknown augmentation kind")
    if "camera_shadow" in kinds and model.model != "pi05":
        raise ValueError("GR00T camera shadows are unsupported")
    if any(kind in kinds for kind in ("shadow_look", "camera_shadow")) and retrieval is None:
        raise ValueError("retrieval fits required for look shadows")
    captured_model = arm.server_meta.get("model", arm.manifest.get("model"))
    if captured_model != model.model:
        raise ValueError("captured/model adapter mismatch")
    if arm.server_meta.get("H") != model.H or arm.server_meta.get("action_dim") != model.action_dim:
        raise ValueError("captured horizon/action dimensions disagree with model")
    if arm.manifest.get("pure_policy"):
        expected = {"current", "bpool_cs" if model.model == "pi05" else "bpool_all"}
        if retrieval is not None and set(retrieval.libraries) != expected:
            raise ValueError("pure-policy shadows require both current and big frozen A libraries")
    aug_dir = Path(aug_dir) if aug_dir is not None else arm.debug_dir / "aug"
    # Legacy fit_config_sha is accepted for API compatibility, but a whole
    # campaign config is never a resume dependency. Each kind hashes used fits.
    provenance = dict(model=model.model, checkpoint_sha=checkpoint_sha, code_sha=code_sha(),
                      dtype="float32", origin="deferred",
                      noise_domain="sha256(campaign|task_id|init|decision_seq|draw); private per-row generators",
                      libraries=getattr(retrieval, "libraries", []), fits=getattr(retrieval, "provenance", {}),
                      fake=isinstance(model, FakeModel) or bool(getattr(model, "fake", False)))
    provenances = {kind: dict(provenance, fit_config_sha=fit_specs_sha(retrieval, kind)) for kind in kinds}
    upgrades = [] if allow_code_upgrade else None
    completed = {kind: _done(aug_dir, kind, provenances[kind], upgrades) for kind in kinds}
    decisions = arm.decisions().sort_values(["task_id", "init", "decision_seq"], kind="stable")
    if not decisions.server_join.eq("verified").all():
        raise ValueError("cannot augment decisions with incomplete client/server joins")
    if limit is not None:
        decisions = decisions.head(limit)
    campaign = arm.manifest.get("campaign", arm.run_root.name)
    upgrade_note = _record_upgrade(aug_dir, upgrades, provenances)
    if upgrade_note:
        for value in provenances.values():
            value["code_upgrade_note"] = upgrade_note
    counts = {kind: 0 for kind in kinds}
    t0 = time.perf_counter()
    previous_profile = getattr(model, "profile", None)
    model.profile = Profile() if profile else None
    try:
        _pipeline(arm, model, retrieval, decisions, campaign, batch_size, kinds, aug_dir,
                  provenances, completed, counts, prefetch)
        model.synchronize()
        timing = model.profile.report() if model.profile else None
    finally:
        model.profile = previous_profile
    seconds = time.perf_counter() - t0
    report = dict(arm=arm.arm_name, newly_written=counts, completed={k: len(v) for k, v in completed.items()},
                  seconds=seconds, decision_rows=len(decisions), requested_batch_size=batch_size,
                  policy_shadow_decisions_per_second=counts.get("policy_shadow", 0) / seconds if seconds else None,
                  provenance=provenance, code_upgrade_note=upgrade_note, profile=timing, prefetch=prefetch,
                  stage_mode=getattr(model, "stage_mode", "batched"))
    _atomic_json(aug_dir / "job_stats.json", report)
    return report


def _pipeline(arm, model, retrieval, decisions, campaign, batch_size, kinds, aug_dir,
              provenances, completed, counts, prefetch):
    # One decoder and one ordered retrieval/publisher. The model stays on the
    # caller's thread. At most one future batch and one output batch are held.
    if "camera_shadow" in kinds:
        with measure(model, "camera_fit_setup"):
            retrieval.ensure_cameras()

    def prepare(records):
        with measure(model, "decode_io"):
            observations = wire_observations(arm, records)
        prepared = model.prepare(observations) if hasattr(model, "prepare") else observations
        return observations, prepared

    def finish(outputs, encoded, records, histories, pending):
        ids = [r["decision_id"] for r in records]
        for kind in kinds:
            selected = [i for i, did in enumerate(ids) if did in pending[kind]]
            if not selected:
                continue
            if kind not in outputs:
                output = dict(decision_id=np.array([ids[i] for i in selected], dtype=schema.DECISION_ID_DTYPE))
                rows = [records[i] for i in selected]
                with measure(model, "retrieval." + kind):
                    if kind == "shadow_look":
                        output.update(retrieval.retrieve(_select(encoded, selected), rows, "full", histories))
                    else:
                        for mode in ("wrist", "third"):
                            camera = retrieval.retrieve(_select(encoded, selected), rows, mode, histories)
                            output.update({mode + "_" + k: v for k, v in camera.items()})
                outputs[kind] = output
            part_provenance = dict(provenances[kind])
            if kind == "camera_shadow":
                part_provenance["fits"] = retrieval.provenance
            with measure(model, "publish_io." + kind):
                _publish(aug_dir, kind, outputs[kind], part_provenance, len(selected))
        return {kind: outputs[kind]["decision_id"].astype(str).tolist() for kind in outputs}

    def accept(result):
        for kind, ids in result.items():
            completed[kind].update(ids)
            counts[kind] += len(ids)

    # Separate single-worker executors avoid transform/retrieval state races.
    with contextlib.ExitStack() as stack:
        decoder = stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="aug-decode")) if prefetch else None
        writer = stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="aug-retrieve")) if prefetch else None
        for _, episode in decisions.groupby("episode_key", sort=False):
            _episode(episode, prepare, finish, accept, decoder, writer, arm, model, campaign,
                     batch_size, kinds, completed)


def _episode(episode, prepare, finish, accept, decoder, writer, arm, model, campaign, batch_size, kinds, completed):
    records_ep = episode.to_dict("records")
    eligible = lambda kind, row: kind != "policy_draws" or schema.draws_sampled(campaign, row["task_uid"], row["decision_seq"])
    pending = {kind: {r["decision_id"] for r in records_ep if eligible(kind, r) and r["decision_id"] not in completed[kind]} for kind in kinds}
    if not any(pending.values()):
        return
    needs_history = any(pending.get(kind) for kind in ("shadow_look", "camera_shadow"))
    records_to_encode = records_ep if needs_history else [r for r in records_ep if any(r["decision_id"] in ids for ids in pending.values())]
    history = dict(v0=[], v1=[], state=[], actions=[], prev_hit=None)
    batches = [records_to_encode[start:start + batch_size] for start in range(0, len(records_to_encode), batch_size)]
    prepared_future = decoder.submit(prepare, batches[0]) if decoder else None
    output_future = None
    for batch_index, records in enumerate(batches):
        observations, prepared = prepared_future.result() if decoder else prepare(records)
        if decoder and batch_index + 1 < len(batches):
            prepared_future = decoder.submit(prepare, batches[batch_index + 1])
        encoded = model.encode_prepared(prepared) if hasattr(model, "encode_prepared") else model.encode(observations)
        ids = [r["decision_id"] for r in records]
        histories = None
        if needs_history:
            with measure(model, "history_io"):
                served = arm.decision_arrays(["served_chunk"], ids)["served_chunk"]
                histories = _history(records, encoded, served, observations, history)
        outputs = {}
        for kind in kinds:
            selected = [i for i, did in enumerate(ids) if did in pending[kind]]
            if not selected:
                continue
            output = dict(decision_id=np.array([ids[i] for i in selected], dtype=schema.DECISION_ID_DTYPE))
            if kind in ("policy_shadow", "policy_draws"):
                # Draws share stage-2 conditioning. Seeds are row-specific
                # so arrival order, batching and resumed parts do not change noise.
                n_draws = 1 if kind == "policy_shadow" else 3
                values, seeds_selected = [], []
                chosen_encoded = model.select(encoded, selected)
                for draw in range(n_draws):
                    domain = 0 if kind == "policy_shadow" else draw + 1
                    seeds = np.array([private_seed(campaign, r["task_id"], r["init"], r["decision_seq"], domain) for r in records], np.int64)
                    # Select existing conditioning; never draw completed
                    # policy rows and never re-encode a sampled subset.
                    model.kind = kind
                    with measure(model, "kind." + kind):
                        chunks = model.draw(chosen_encoded, seeds[selected])
                    if chunks.shape != (len(selected), model.H, model.action_dim) or not np.isfinite(chunks).all():
                        raise ValueError("malformed/nonfinite shadow chunks")
                    values.append(chunks.astype(np.float32))
                    seeds_selected.append(seeds[selected])
                if kind == "policy_shadow":
                    output.update(chunk=values[0], seed=seeds_selected[0],
                                  input_sha=np.array([input_sha(observations[i]) for i in selected], dtype="<U64"))
                else:
                    output.update(chunks=np.stack(values, axis=1), seeds=np.stack(seeds_selected, axis=1))
                outputs[kind] = output
        # Strip GPU conditioning before handing keys to the CPU worker.
        cpu_keys = _select(encoded, list(range(len(records))))
        if output_future is not None:
            accept(output_future.result())
        if writer:
            output_future = writer.submit(finish, outputs, cpu_keys, records, histories, pending)
        else:
            accept(finish(outputs, cpu_keys, records, histories, pending))
    if output_future is not None:
        accept(output_future.result())


def benchmark(arm, model, batch_sizes=(1, 8, 32), limit=96):
    previous = arm.cache_enabled
    arm.cache_enabled = False
    try:
        return _benchmark(arm, model, batch_sizes, limit)
    finally:
        arm.cache_enabled = previous


def _benchmark(arm, model, batch_sizes=(1, 8, 32), limit=96):
    records = arm.decisions().head(limit).to_dict("records")
    if not records:
        raise ValueError("no decisions for benchmark")
    observations = wire_observations(arm, records)
    campaign = arm.manifest.get("campaign", arm.run_root.name)
    results = []
    warmup = model.encode(observations[:1])
    model.draw(warmup, [private_seed(campaign, records[0]["task_id"], records[0]["init"], records[0]["decision_seq"], 0)])
    model.synchronize()
    for size in batch_sizes:
        model.synchronize()
        t0 = time.perf_counter()
        for start in range(0, len(records), size):
            rows = records[start:start + size]
            encoded = model.encode(observations[start:start + size])
            model.draw(encoded, [private_seed(campaign, r["task_id"], r["init"], r["decision_seq"], 0) for r in rows])
        model.synchronize()
        elapsed = time.perf_counter() - t0
        results.append(dict(model=model.model, batch_size=size, decisions=len(records), seconds=elapsed,
                            decisions_per_second=len(records) / elapsed,
                            batching_scope="all pi05 stages; eager same-shape GR00T stages 1/2/3",
                            stage_mode=getattr(model, "stage_mode", "batched")))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arms", nargs="+", required=True, help="priority order; P0 first")
    parser.add_argument("--model", choices=("pi05", "groot"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kinds", nargs="+", choices=KINDS)
    parser.add_argument("--fit-config", type=Path, help="JSON mapping arm name to frozen fit specifications")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fake", action="store_true", help="CPU fixture use only")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--stage-mode", choices=("serial", "batched"), default="batched", help="GR00T encoder/LLM reference or batched path")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--profile", action="store_true", help="wall timers plus CUDA events; synchronize once at job end")
    parser.add_argument("--allow-code-upgrade", action="store_true", help="record old part hashes, preserve their bytes, permit only code-SHA differences")
    parser.add_argument("--output-root", type=Path, help="write aug outputs under ROOT/ARM; read captured run without modifying it")
    parser.add_argument("--benchmark-all", action="store_true", help="all requested kinds including decode/retrieval/publication, fresh --output-root required")
    args = parser.parse_args()
    if not args.fake and not args.checkpoint:
        parser.error("--checkpoint required for real model")
    if args.benchmark_all and (args.output_root is None or args.benchmark):
        parser.error("--benchmark-all requires --output-root and excludes --benchmark")
    model = FakeModel(args.model) if args.fake else (Pi05Model(args.checkpoint, args.device) if args.model == "pi05" else GrootModel(args.checkpoint, args.device, stage_mode=args.stage_mode))
    config = reader.read_json(args.fit_config) if args.fit_config else {}
    for name in args.arms:
        arm = reader.open_arm(args.run_root, name)
        arm.cache_enabled = False
        output_dir = args.output_root / name if args.output_root else arm.debug_dir / "aug"
        if args.benchmark:
            report = benchmark(arm, model, args.batch_sizes, args.limit or 96)
            target = output_dir / "benchmark.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n")
        else:
            need_fit = args.kinds is None or any(kind in args.kinds for kind in ("shadow_look", "camera_shadow"))
            retrieval = (FakeRetrieval(args.model) if args.fake else FrozenRetrieval(config[name], output_dir, args.model, args.kinds)) if need_fit else None
            if args.benchmark_all:
                sizes = args.batch_sizes
                targets = [output_dir / ("benchmark_b%d" % size) for size in sizes]
                if len(set(sizes)) != len(sizes) or any(size < 1 for size in sizes):
                    parser.error("benchmark batch sizes must be distinct and positive")
                if any(list(target.glob("*/*.npz")) for target in targets):
                    raise ValueError("benchmark output already contains parts; use a fresh --output-root")
                if args.model == "pi05" and retrieval is not None and (args.kinds is None or "camera_shadow" in args.kinds):
                    retrieval.ensure_cameras()
                # Warm one policy decision outside timed runs. Each run then
                # includes its own decode, transform, retrieval and atomic I/O.
                rows = arm.decisions().head(1).to_dict("records")
                if not rows:
                    raise ValueError("no decisions for benchmark")
                encoded = model.encode(wire_observations(arm, rows))
                model.draw(encoded, [private_seed(arm.manifest.get("campaign", arm.run_root.name), rows[0]["task_id"], rows[0]["init"], rows[0]["decision_seq"], 0)])
                model.synchronize()
                report = [run_arm(arm, model, retrieval, size, args.kinds, checkpoint_sha=args.checkpoint_sha,
                                  limit=args.limit or 96, prefetch=not args.no_prefetch, profile=True,
                                  output_dir=target) for size, target in zip(sizes, targets)]
                _atomic_json(output_dir / "benchmark_all.json", report)
            else:
                report = run_arm(arm, model, retrieval, args.batch_size, args.kinds, checkpoint_sha=args.checkpoint_sha,
                                 limit=args.limit, allow_code_upgrade=args.allow_code_upgrade,
                                 prefetch=not args.no_prefetch, profile=args.profile, output_dir=output_dir)
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
