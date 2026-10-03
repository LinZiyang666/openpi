"""Library physics backfill, admitted only with exact executed-control provenance.

Store action.npy is a normalized proposal, never an executed control log.
Missing resets/actions are reported rather than replaced by guessed prefixes.
The coordinator supplies a source map when those records become available.
"""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exp.offline_search.debug.client.capture import Client, EnvTap, FileSink
from exp.offline_search.debug.transport.receiver import atomic_json, digest


def robot_state(obs):
    quat = np.array(obs["robot0_eef_quat"], dtype=np.float64, copy=True)
    den = np.sqrt(max(0., 1. - float(np.clip(quat[3], -1., 1.)) ** 2))
    axis = np.zeros(3) if den < 1e-8 else quat[:3] * (2 * np.arccos(np.clip(quat[3], -1., 1.)) / den)
    return np.concatenate((obs["robot0_eef_pos"], axis, obs["robot0_gripper_qpos"]))


def inspect_library(library, sources=None):
    """Metadata-only inventory; never reads tokens, policy or action proposals."""
    library = Path(library)
    episodes = json.loads((library / "episodes.json").read_text())
    mappings = sources or {}
    rows = []
    for i, ep in enumerate(episodes):
        source = mappings.get(str(i), mappings.get(ep["stem"]))
        rows.append(dict(episode=i, stem=ep["stem"], task_id=ep["task_id"], library_source=ep["file"],
                         orig_init_state_idx=ep.get("orig_init_state_idx"), source=source,
                         status="available" if source else "unsupported",
                         reason="explicit reset/control source supplied; replay validation pending" if source else
                         "store has rs and normalized action proposals, not initial simulator state or executed controls"))
    return dict(library=str(library.resolve()), library_manifest_sha256=digest(library / "manifest.json"),
                episodes=len(rows), replay_candidates=sum(r["source"] is not None for r in rows),
                certified_replays=0, records=rows)


def load_source(source, base=None):
    """Source NPZ: initial_state, action, decision_seq; settle controls use -1.

    decision_seq is the original library decision/step, not synthesized groups.
    Each returned row records the exact action passed to env.step. Initial state
    and seed must be from the same collection, not an A-pool with matching IDs.
    """
    path = Path(source["npz"])
    if base and not path.is_absolute():
        path = Path(base) / path
    if not source.get("provenance") or "env_seed" not in source or "sha256" not in source:
        raise ValueError("source requires provenance, env_seed and sha256")
    if digest(path) != source["sha256"]:
        raise ValueError("source payload hash mismatch")
    with np.load(str(path), allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    if any(k not in result for k in ("initial_state", "action", "decision_seq")):
        raise ValueError("source missing reset/executed actions/decision assignment")
    action, seq = result["action"], result["decision_seq"]
    if action.ndim != 2 or seq.shape != (len(action),) or seq.dtype.kind not in "iu" or not np.isfinite(action).all():
        raise ValueError("invalid replay source shapes/types")
    if len(seq) == 0 or (seq < -1).any() or np.any(np.diff(seq) < 0):
        raise ValueError("source decision sequence is empty/noncontiguous")
    unique = np.unique(seq[seq >= 0])
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("source decisions must include the complete prefix from zero")
    return result


class ReplayClient:
    def episode_start(self, **kw):
        pass

    def infer(self, obs):
        did = obs["__debug__"]["decision_id"]
        return {"actions": self.chunk, "__debug__": {"v": 1, "decision_id": did, "status": "available", "origin": "library_replay"}}


def replay_episode(env, source, source_meta, ep, library_rows, expected_rs, out, arm="library", atol=1e-6, state_transform=None):
    """No model import/call. Failed state/outcome validation retains all records."""
    inner = ReplayClient()
    client = Client(inner, out, campaign="library_backfill", arm=arm, env_seed=int(source_meta["env_seed"]), sink_factory=FileSink)
    client.episode_start(experiment=source_meta["suite"], task=ep["task"], episode_id=ep.get("orig_init_state_idx"),
                         extra_metadata=dict(task_uid=arm + ":library:" + str(ep["task_id"]) + ":" + str(source_meta["episode"]),
                                             attempt=int(source_meta.get("attempt", 1)), task_id=ep["task_id"],
                                             orig_init_state_idx=ep.get("orig_init_state_idx")))
    settle = int(np.sum(source["decision_seq"] == -1))
    client.configure(env, SimpleNamespace(num_steps_wait=settle, replan_steps=1, seed=source_meta["env_seed"]), len(source["action"]) - settle)
    client.episode["backfill"] = dict(provenance=source_meta["provenance"], source_sha256=source_meta["sha256"],
                                     library_rows=[int(x) for x in library_rows], admission="pending")
    tap = EnvTap(env, client)
    differences = []
    sim_differences = []
    done = False
    try:
        env.reset()
        obs = tap.set_init_state(source["initial_state"])
        previous = -1
        for i, (action, seq) in enumerate(zip(source["action"], source["decision_seq"])):
            seq = int(seq)
            if seq != previous and seq >= 0:
                wire_state = robot_state(obs)
                if "expected_robot_state_wire" in source and not np.allclose(wire_state, source["expected_robot_state_wire"][seq], rtol=0., atol=atol):
                    raise ValueError("robot wire state differs from exact trajectory at decision " + str(seq))
                if "expected_sim_state" in source:
                    actual_sim = np.asarray(env.get_sim_state(), dtype=np.float64)
                    target_sim = source["expected_sim_state"][seq]
                    if actual_sim.shape != target_sim.shape:
                        raise ValueError("simulator state shape differs from exact trajectory")
                    sim_error = float(np.max(np.abs(actual_sim - target_sim)))
                    sim_differences.append(sim_error)
                    if not np.isfinite(sim_error) or sim_error > atol:
                        raise ValueError("simulator state differs at decision {}: {}".format(seq, sim_error))
                actual = np.asarray(state_transform(wire_state) if state_transform else wire_state, dtype=np.float64)
                if seq >= len(expected_rs):
                    raise ValueError("source has more decisions than library episode")
                target = np.asarray(expected_rs[seq])
                # A declared transform returns the full model-state shape;
                # wire-space sources compare valid channels only (no padding).
                if state_transform is None:
                    target = target[:len(actual)]
                if target.shape != actual.shape:
                    raise ValueError("robot-state adapter shape mismatch")
                error = float(np.max(np.abs(actual - target)))
                differences.append(error)
                if not np.isfinite(error) or error > atol:
                    raise ValueError("robot-state mismatch at library decision {}: {} > {}".format(seq, error, atol))
                mask = source["decision_seq"] == seq
                inner.chunk = np.array(source["action"][mask], copy=True)
                client.infer({"observation/state": wire_state, "prompt": ep["task"]})
            obs, reward, done, info = tap.step(action.tolist())
            previous = seq
            if done and i != len(source["action"]) - 1:
                raise ValueError("replay terminates before source controls end")
        if len(differences) != len(expected_rs):
            raise ValueError("source does not cover every library decision")
        if bool(done) != bool(ep["success"]):
            raise ValueError("replay success differs from library")
        if client.errors or any(v.get("status") == "error" for v in client.adapter.capabilities.values()):
            raise ValueError("capture errors or capability failure before backfill admission")
        client.episode["backfill"].update(admission="PASS", max_abs_rs_error=max(differences, default=0.), atol=atol,
                                         max_abs_sim_error=max(sim_differences) if sim_differences else None,
                                         source_metadata=source_meta)
        client.finish(done, "success" if done else "source_end")
        if client.errors:
            raise ValueError("capture errors: " + str(client.errors))
        return dict(status="PASS", episode_key=client.identity["episode_key"], max_abs_rs_error=max(differences, default=0.), success=bool(done))
    except Exception as exc:
        client.episode["backfill"].update(admission="REJECTED", reason=str(exc), rs_errors=differences)
        client.finish(done, "backfill_rejected", str(exc))
        return dict(status="REJECTED", episode_key=client.identity["episode_key"], reason=str(exc))


def libero_factory(meta):
    from exp.offline_search.debug.client.compat import install_import_compat
    install_import_compat()
    from libero.libero import benchmark
    from examples.libero import main
    task = benchmark.get_benchmark_dict()[meta["suite"]]().get_task(int(meta["task_id"]))
    env, _ = main._get_libero_env(task, main.LIBERO_ENV_RESOLUTION, int(meta["env_seed"]))
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--source-map", type=Path, help="JSON mapping episode index/stem -> {npz,sha256,env_seed,suite,provenance}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay", action="store_true", help="coordinator simulator run; default inventory only")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--env-factory", default="exp.offline_search.debug.backfill.replay:libero_factory")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--state-transform", help="CPU module:callable mapping wire state -> full stored rs; required for model-normalized stores")
    ap.add_argument("--trajectory-root", type=Path, help="original collection save_traj root, containing suite/task_N/episode_M.h5")
    ap.add_argument("--suite", help="benchmark suite for --trajectory-root")
    ap.add_argument("--init-states-dir", help="original B-pool directory containing task-name .init files")
    ap.add_argument("--norm-stats", type=Path, help="pi05 LIBERO state quantile artifact; no model import")
    a = ap.parse_args()
    sources = json.loads(a.source_map.read_text()) if a.source_map else {}
    if a.trajectory_root:
        if not a.suite:
            ap.error("--suite required with --trajectory-root")
        from .trajectories import discover
        for key, value in discover(a.library, a.trajectory_root, a.suite).items():
            sources.setdefault(key, value)
    report = inspect_library(a.library, sources)
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out / "inventory.json", report)
    if a.replay:
        module, name = a.env_factory.split(":")
        factory = getattr(importlib.import_module(module), name)
        state_transform = None
        if a.state_transform:
            state_module, state_name = a.state_transform.split(":")
            state_transform = getattr(importlib.import_module(state_module), state_name)
        eps = json.loads((a.library / "episodes.json").read_text())
        rs = np.load(str(a.library / "rs.npy"), mmap_mode="r", allow_pickle=False)
        if a.norm_stats:
            if a.state_transform:
                ap.error("choose --norm-stats or --state-transform")
            from .trajectories import normalization
            state_transform = normalization(a.norm_stats, rs.shape[1])
        episode_idx = np.load(str(a.library / "episode.npy"), mmap_mode="r", allow_pickle=False)
        steps = np.load(str(a.library / "step.npy"), mmap_mode="r", allow_pickle=False)
        results = []
        for row in report["records"]:
            i = row["episode"]
            if a.episodes is not None and i not in a.episodes:
                continue
            if not row["source"]:
                results.append(dict(episode=i, status="UNSUPPORTED", reason=row["reason"]))
                continue
            try:
                meta = dict(row["source"], episode=i, task_id=eps[i]["task_id"])
                if "h5" in meta:
                    from .trajectories import initial_from_pool, load
                    initial = initial_from_pool(meta, eps[i], a.init_states_dir)
                    source, loaded_meta = load(meta["h5"], initial, eps[i], meta["suite"])
                    meta.update(loaded_meta)
                    if a.norm_stats:
                        meta["state_norm_sha256"] = digest(a.norm_stats)
                else:
                    source = load_source(row["source"], a.source_map.parent if a.source_map else None)
                if state_transform is None and meta.get("rs_space") != "wire":
                    raise ValueError("declare rs_space=wire or supply --state-transform for model-normalized library rs")
                rows = np.flatnonzero(episode_idx == i)
                if not np.array_equal(steps[rows], np.arange(len(rows))):
                    raise ValueError("library rows are not a complete contiguous episode")
                if "expected_robot_state_wire" in source and state_transform:
                    recorded_rs = np.asarray([state_transform(w) for w in source["expected_robot_state_wire"]])
                    if recorded_rs.shape != rs[rows].shape or not np.allclose(recorded_rs, rs[rows], rtol=0., atol=a.atol):
                        raise ValueError("save_traj does not match library rs through the declared normalization")
                env = factory(meta)
                try:
                    result = replay_episode(env, source, meta, eps[i], rows, rs[rows], a.out / "client", atol=a.atol, state_transform=state_transform)
                finally:
                    env.close()
                results.append(dict(episode=i, **result))
            except Exception as exc:
                results.append(dict(episode=i, status="REJECTED", reason=str(exc)))
            atomic_json(a.out / "replay_report.json", results)
        atomic_json(a.out / "replay_report.json", results)
        report = dict(replayed=len(results), admitted=sum(r["status"] == "PASS" for r in results))
        if any(r["status"] != "PASS" for r in results):
            print(json.dumps(report))
            raise SystemExit(1)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
