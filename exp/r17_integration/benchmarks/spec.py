"""CPU-only, versioned benchmark contracts. No simulator imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Protocol
from pathlib import Path


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()


def canonical_table_identity(spec) -> str:
    """P1's r17.task_table.v1 identity without import-time registry mutation."""
    from exp.r17_integration.common.contracts import digest as contract_digest
    contract = spec.to_contract()
    return contract_digest({"schema": "r17.task_table.v1", "benchmark_identity": contract.identity,
                            "protocol_identity": contract.identity, "ordered_tasks": contract.ordered_tasks})


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    protocol_id: str
    tasks: tuple[tuple[str, str, str], ...]
    seed: int = 7
    settle_steps: int = 15
    max_policy_controls: int = 160
    replan_steps: int = 5
    camera: str = "corner2"
    camera_id: int = 2
    camera_pos: tuple[float, ...] = (0.75, 0.075, 0.7)
    render_size: int = 480
    rotation_degrees: int = 180
    formal_indices: tuple[int, ...] = tuple(range(50))
    library_indices: tuple[int, ...] = tuple(range(50))
    dev_indices: tuple[int, ...] = ()
    library_seed: int = 8
    dev_seed: int | None = None
    interface_version: str = "r17.v1"

    def seed_for(self, partition: str) -> int:
        seed = {"test": self.seed, "smoke": self.seed,
                "library": self.library_seed, "dev": self.dev_seed}[partition]
        if seed is None:
            raise ValueError("dev partition seed has not been chosen")
        return seed

    @property
    def ordered_tasks(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self.tasks)

    @property
    def task_table_identity(self) -> str:
        return canonical_table_identity(self)

    @property
    def identity(self) -> str:
        return self.to_contract().identity

    def to_contract(self):
        from exp.r17_integration.common.contracts import BenchmarkSpec as Contract
        return Contract(self.benchmark, self.interface_version, self.protocol_id,
                        tuple(r[0] for r in self.tasks), tuple(r[1] for r in self.tasks),
                        self.replan_steps, self.settle_steps, self.max_policy_controls,
                        camera_digest=digest((self.camera, self.camera_id, self.camera_pos,
                                              self.render_size, self.rotation_degrees)),
                        group_digest=digest((self.tasks, self.seed, self.formal_indices,
                                             self.library_indices, self.dev_indices,
                                             self.library_seed, self.dev_seed)),
                        env_source_digest=digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                                  (Path(__file__).resolve().parents[2] / "metaworld/env.py",
                                                   Path(__file__).resolve().parents[2] / "metaworld/tasks.py")}),
                        success_semantics="info.success at any policy control",
                        termination_semantics="success | terminated | truncated | step_cap; exception is error")

    def manifest(self) -> dict:
        return {**asdict(self), "protocol_identity": self.identity,
                "ordered_tasks": self.ordered_tasks, "task_table_identity": self.task_table_identity,
                "expected_formal_episodes": len(self.tasks) * len(self.formal_indices),
                "init_source": "MT1(task, seed=partition_seed).train_tasks[original_idx]",
                "success": "info.success at any policy control",
                "policy_rng": "sha256(mt50:episode_seed:task:idx:decision)[:4] little-endian; NumPy PCG64 standard_normal float32",
                "termination": "success | terminated | truncated | step_cap; exceptions are errors"}


class PhysicalAdapter(Protocol):
    def capture(self, obs=None, info=None, requested_action=None, applied_action=None) -> dict: ...


class BenchmarkAdapter(Protocol):
    def tasks(self) -> tuple: ...
    def episode_specs(self, partition: str = "test") -> list[dict]: ...
    def make_episode_runner(self, client_factory, capture_sink=None, execution_ack=None): ...
