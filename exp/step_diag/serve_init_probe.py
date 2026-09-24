"""Run initialization counterfactuals on fixed teacher-visited RoboCasa observations.

This experiment-only wrapper composes the existing diagnostic servers. The full
teacher still controls the environment. At selected decisions, clean retrieved
actions, another trajectory's same-task actions, zeros, and private Gaussian
noise initialize fresh one- and two-step loops. All loop states and observations
are saved, and auxiliary calls preserve the teacher's random stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def stable_seed(*parts) -> int:
    """Derive a private reproducible seed without consuming a global RNG."""
    raw = json.dumps(parts, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def chunk_numpy(value) -> np.ndarray:
    """Copy an unbatched float32 chunk to CPU for immutable evidence."""
    a = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"invalid action chunk: {a.shape}")
    return np.array(a, dtype=np.float32, copy=True)


def action_rmse(a, b) -> float:
    """RMSE on the executed first five actions and twelve active coordinates."""
    delta = np.asarray(a, dtype=np.float64)[:5, :12] - np.asarray(b, dtype=np.float64)[:5, :12]
    return float(np.sqrt(np.mean(delta**2)))


def control_entry_ids(entries, winner_id: str) -> list[str]:
    """Select same-task controls from trajectories other than the winner's."""
    winner = entries[winner_id]
    if not winner.payload.task_key or not winner.trajectory_id:
        raise ValueError("winner lacks task or trajectory identity")
    return sorted(eid for eid, entry in entries.items()
                  if entry.payload.task_key == winner.payload.task_key
                  and entry.trajectory_id and entry.trajectory_id != winner.trajectory_id)


class InitProbe:
    """Persist paired counterfactuals and their numerical integrity checks."""

    def __init__(self, out: Path, policy: str, decisions: tuple[int, ...]):
        self.out = out
        self.policy = policy
        self.decisions = decisions
        self.pools = {}
        out.mkdir(parents=True, exist_ok=True)

    def record(self, *, episode, idx, storage, winner_id, teacher, run_loop, observation=None):
        """Evaluate four initializations at both budgets on a shared condition."""
        entries = storage._backend._entries
        winner = storage.fetch_entry(winner_id)
        key = (id(storage), winner.payload.task_key, winner.trajectory_id)
        if key not in self.pools:
            self.pools[key] = control_entry_ids(entries, winner_id)
        candidates = self.pools[key]
        if not candidates:
            raise ValueError("no independent same-task cache trajectory")
        seed = stable_seed(self.policy, episode.task, episode.env_seed, episode.init_idx, idx)
        rng = np.random.default_rng(seed)
        random_id = candidates[int(rng.integers(len(candidates)))]
        cached = chunk_numpy(winner.payload.action_chunk)
        random_cached = chunk_numpy(storage.fetch_payload(random_id).action_chunk)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(cached.shape, generator=generator).numpy()
        starts = {"retrieved": cached, "random_same_task": random_cached,
                  "zero": np.zeros_like(cached), "gaussian": noise}
        arrays = {"teacher": chunk_numpy(teacher)}
        if observation is not None:
            for name, value in observation.items():
                arr = np.asarray(value)
                if arr.dtype.kind != "O":
                    arrays[f"obs__{name}"] = arr
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        metrics = []
        with torch.random.fork_rng(), torch.no_grad():
            for n in (1, 2):
                trajectories = {}
                for name, start in starts.items():
                    before = start.copy()
                    path = np.stack([chunk_numpy(x) for x in run_loop(torch.from_numpy(start.copy()), n)])
                    if path.shape != (n + 1, *cached.shape) or not np.isfinite(path).all():
                        raise ValueError("counterfactual loop shape or finiteness failure")
                    if not np.array_equal(start, before):
                        raise ValueError("initialization mutated")
                    trajectories[name] = path
                    arrays[f"{name}_n{n}"] = path
                repeated = np.stack([chunk_numpy(x) for x in run_loop(torch.from_numpy(cached.copy()), n)])
                repeat_max_abs = float(np.max(np.abs(repeated - trajectories["retrieved"])))
                if repeat_max_abs > 1e-5:
                    raise ValueError(f"identical-condition repeat differs: {repeat_max_abs}")
                for control in ("random_same_task", "zero", "gaussian"):
                    a, b = trajectories["retrieved"], trajectories[control]
                    initial = action_rmse(a[0], b[0])
                    first = action_rmse(a[1], b[1])
                    final = action_rmse(a[-1], b[-1])
                    metrics.append({"n": n, "control": control, "input_rmse": initial,
                                    "first_rmse": first, "final_rmse": final,
                                    "first_ratio": first / initial if initial > 1e-12 else None,
                                    "final_ratio": final / initial if initial > 1e-12 else None,
                                    "repeat_max_abs": repeat_max_abs})
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise ValueError("probe changed CPU RNG")
        if any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
            raise ValueError("probe changed CUDA RNG")
        if not np.array_equal(cached, chunk_numpy(winner.payload.action_chunk)):
            raise ValueError("cache payload mutated")
        tag = hashlib.sha256(f"{episode.task_uid}:{episode.attempt}:{idx}".encode()).hexdigest()[:24]
        target = self.out / f"arrays_{tag}.npz"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        np.savez_compressed(target, **arrays)
        row = {"schema": 1, "policy": self.policy, "task": episode.task, "task_uid": episode.task_uid,
               "attempt": episode.attempt, "env_seed": episode.env_seed, "init_idx": episode.init_idx,
               "decision_idx": idx, "seed": seed, "winner_id": winner_id,
               "winner_task": winner.payload.task_key, "winner_trajectory": winner.trajectory_id,
               "random_id": random_id, "random_trajectory": entries[random_id].trajectory_id,
               "random_task": entries[random_id].payload.task_key, "control_pool_size": len(candidates),
               "cache_noise_T": 0.0, "native_start_time": 0.0 if self.policy == "groot" else 1.0,
               "arrays": target.name, "arrays_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "rng_preserved": True, "payload_preserved": True, "metrics": metrics}
        with (self.out / "probe.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"INIT_PROBE task={episode.task} seed={episode.env_seed} decision={idx} arrays={target.name}", flush=True)


def install_groot(probe: InitProbe) -> None:
    """Replace auxiliary diagnostics while keeping the full executed policy."""
    from exp.step_diag.groot import GrootDiagPolicy
    from openpi.cache.groot import staged

    original_get = GrootDiagPolicy.get_action

    def get_action(self, observations):
        self._probe_observation = observations
        try:
            return original_get(self, observations)
        finally:
            self._probe_observation = None

    def shadow(self, cp1, stage2, teacher, action_cpu, *, retrieval_error=None):
        episode, idx = self._diag._episode, self._diag._decision_idx
        self._diag.record(a_exec=action_cpu, executed_steps=teacher.steps_run,
                          n_stage3_calls=1, hit_type="MISS", start_t=None,
                          schedule_id=self._schedule.schedule_id)
        if episode is None or idx not in probe.decisions:
            return
        if retrieval_error or cp1 is None or not cp1.entry_id:
            raise ValueError(f"missing retrieval: {retrieval_error}")
        runner = self._runner

        def run_loop(start, n):
            path = []

            def on_step(i, x_in, x_out):
                if i == 0:
                    path.append(chunk_numpy(x_in))
                path.append(chunk_numpy(x_out))

            with runner.session():
                staged.denoise_loop(runner._model.action_head, runner._head_inputs(stage2),
                                    stage2.action_inputs, noise=start[None], num_steps=n, on_step=on_step)
            return path

        probe.record(episode=episode, idx=idx, storage=self._orchestrator._storage,
                     winner_id=cp1.entry_id, teacher=action_cpu, run_loop=run_loop,
                     observation=self._probe_observation)

    GrootDiagPolicy.get_action = get_action
    GrootDiagPolicy._shadow = shadow


def install_pi05(probe: InitProbe) -> None:
    """Observe the existing pi0.5 stage-2 handle without altering its teacher arm."""
    from exp.step_diag.pi05 import Pi05DiagInterceptor, warm_variant_stage3

    original_get = Pi05DiagInterceptor.infer
    original_record = Pi05DiagInterceptor._record_shadow_teacher_arm

    def infer(self, obs, *, noise=None):
        self._probe_observation = obs
        try:
            return original_get(self, obs, noise=noise)
        finally:
            self._probe_observation = None

    def record(self, cp1_result, stage3):
        episode, idx = self._diag._episode, self._diag._decision_idx
        old_mode = self._diag_mode
        self._diag_mode = "full"
        try:
            original_record(self, cp1_result, stage3)
        finally:
            self._diag_mode = old_mode
        if episode is None or idx not in probe.decisions:
            return
        results = self._search_rec.last_results
        if not results:
            raise ValueError("missing pi0.5 retrieval")
        stage2 = self._tl.stage2

        def run_loop(start, n):
            model = self._model
            original_step = model.denoise_step
            path = []

            def step(state, masks, kv, x, t):
                v = original_step(state, masks, kv, x, t)
                if not path:
                    path.append(chunk_numpy(x))
                path.append(chunk_numpy(x - v / n))
                return v

            model.denoise_step = step
            try:
                result = warm_variant_stage3(model, stage2, start.to(self._stage3_device)[None],
                                             n / 10, num_steps=10, variant="reset_final")
                if not np.allclose(path[-1], chunk_numpy(result.action_chunk), atol=1e-6, rtol=0):
                    raise ValueError("captured pi0.5 loop differs from executed loop")
            finally:
                model.denoise_step = original_step
            return path

        self._tl.in_shadow = True
        try:
            probe.record(episode=episode, idx=idx, storage=self._orchestrator._storage,
                         winner_id=results[0].id, teacher=stage3.action_chunk,
                         run_loop=run_loop, observation=self._probe_observation)
        finally:
            self._tl.in_shadow = False

    Pi05DiagInterceptor.infer = infer
    Pi05DiagInterceptor._record_shadow_teacher_arm = record


def main() -> None:
    """Start a dedicated probe server using the existing server CLI after ``--``."""
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("groot", "pi05"), required=True)
    parser.add_argument("--probe-out", type=Path, required=True)
    parser.add_argument("--decisions", default="0,5,15,30,60")
    args = parser.parse_args(sys.argv[1:split])
    rest = sys.argv[split + 1:]
    decisions = tuple(int(i) for i in args.decisions.split(","))
    probe = InitProbe(args.probe_out, args.policy, decisions)
    contract = {"schema": 1, "policy": args.policy, "decisions": decisions, "N": [1, 2], "T": 0,
                "executed": "full teacher", "metric": "RMSE first 5 actions, first 12 active normalized coordinates",
                "random_control": "uniform entry from winner task, excluding winner trajectory",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "argv": rest}
    contract_path = args.probe_out / "probe_contract.json"
    if contract_path.exists():
        raise FileExistsError(f"use a fresh run directory: {contract_path}")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    from exp.step_diag import envs

    envs.git_commit = lambda *a, **kw: "workspace-source-hashes-in-runtime-manifest"
    if args.policy == "groot":
        install_groot(probe)
        from exp.step_diag.serve_diag_groot import main as serve
    else:
        install_pi05(probe)
        from exp.step_diag.serve_diag_pi05 import main as serve
    serve(rest)


if __name__ == "__main__":
    main()
