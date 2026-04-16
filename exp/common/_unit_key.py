"""Unit-key dataclasses for experiment runners (cleanup/03a, plan P0-5).

Each runner persists state JSON keyed by a colon-joined tuple. Four
pre-cleanup copies of the serialization pattern existed
(``run_step1b_gt.py``, Phase1 / Phase2 in ``compute_deviate_scores.py``,
and ``run_spawn_experiment.py``) with subtly different field orders and
escape semantics, making the JSON format testable only at call sites.

We deliberately **do not** collapse the four into one ``UnitKey``: the
first field of Step 1b is ``task_id`` (int), the first field of
Phase/Spawn is ``cfg`` (str). A single dataclass would either lose
field semantics or force the callers to pass placeholders. Instead each
runner gets its own frozen dataclass with ``encode`` / ``decode``.

Format contract (byte-equal to pre-cleanup strings):

    Step1bKey(task_id=3, init_idx=17)
        -> "3:17"

    DeviateKey(cfg="clip_w7_d4", ep="task_0/episode_1", sample_idx=5)
        -> "clip_w7_d4:task_0/episode_1:5"           # Phase1

    DeviateKey(cfg="clip_w7_d4", ep="task_0/episode_1", sample_idx=None)
        -> "clip_w7_d4:task_0/episode_1"             # Phase2

    SpawnKey(cfg="clip_w7_d4", ep="task_0/episode_0",
             s=12, n=1, k_idx=0)
        -> "clip_w7_d4:task_0/episode_0:s12:n1:k0"

``ep`` contains a ``/`` (``task_X/episode_Y``) but **no** ``:``, which is
why colon-split parsing is unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Step1bKey:
    """Key for ``exp/trajectory_deviation/run_step1b_gt.py`` state JSON."""

    task_id: int
    init_idx: int

    def encode(self) -> str:
        return f"{int(self.task_id)}:{int(self.init_idx)}"

    @classmethod
    def decode(cls, key: str) -> "Step1bKey":
        parts = key.split(":")
        if len(parts) != 2:
            raise ValueError(f"Malformed Step1bKey: {key!r}")
        try:
            return cls(task_id=int(parts[0]), init_idx=int(parts[1]))
        except ValueError as e:
            raise ValueError(f"Malformed Step1bKey: {key!r}") from e


@dataclass(frozen=True)
class DeviateKey:
    """Key for ``exp/trajectory_deviation/compute_deviate_scores.py`` Phase1 / Phase2.

    Phase1 carries a per-episode stochastic sample index; Phase2 has
    exactly one unit per episode and uses ``sample_idx=None``.
    """

    cfg: str
    ep: str
    sample_idx: int | None = None

    def encode(self) -> str:
        if self.sample_idx is None:
            return f"{self.cfg}:{self.ep}"
        return f"{self.cfg}:{self.ep}:{int(self.sample_idx)}"

    @classmethod
    def decode(cls, key: str) -> "DeviateKey":
        # Phase1 has 3 colon-separated segments where the second contains
        # a ``/``; Phase2 has 2. The ``ep`` never contains ``:``, so a
        # bounded split disambiguates unambiguously.
        parts = key.split(":")
        if len(parts) == 2:
            cfg, ep = parts
            if not cfg or not ep:
                raise ValueError(f"Malformed DeviateKey: {key!r}")
            return cls(cfg=cfg, ep=ep, sample_idx=None)
        if len(parts) == 3:
            cfg, ep, s_str = parts
            if not cfg or not ep:
                raise ValueError(f"Malformed DeviateKey: {key!r}")
            try:
                return cls(cfg=cfg, ep=ep, sample_idx=int(s_str))
            except ValueError as e:
                raise ValueError(f"Malformed DeviateKey: {key!r}") from e
        raise ValueError(f"Malformed DeviateKey: {key!r}")


_SPAWN_KEY_RE = re.compile(
    r"^(?P<cfg>[^:]+):(?P<ep>[^:]+):s(?P<s>\d+):n(?P<n>\d+):k(?P<k>\d+)$"
)


_STEP3_KEY_RE = re.compile(
    r"^(?P<cfg>[^:]+):(?P<ep>[^:]+):tau(?P<tau>\d+):n(?P<n>\d+)$"
)


@dataclass(frozen=True)
class Step3PerCycleKey:
    """Key for ``exp/trajectory_deviation/run_step3_per_cycle_policy.py``.

    Format: ``<cfg>:task_<task_id>/episode_<orig_init_idx>:tau<tau>:n<n>``.
    ``tau`` / ``n`` are stored as non-negative integers (the plan's sweep
    grid uses ``tau in {3, 5, 7, 10}`` and ``n in {1, 2, 3, 5, 10}``).
    """

    cfg: str
    ep: str
    tau: int
    n: int

    def encode(self) -> str:
        return f"{self.cfg}:{self.ep}:tau{int(self.tau)}:n{int(self.n)}"

    @classmethod
    def decode(cls, key: str) -> "Step3PerCycleKey":
        m = _STEP3_KEY_RE.match(key)
        if not m:
            raise ValueError(f"Malformed Step3PerCycleKey: {key!r}")
        return cls(
            cfg=m.group("cfg"),
            ep=m.group("ep"),
            tau=int(m.group("tau")),
            n=int(m.group("n")),
        )


@dataclass(frozen=True)
class SpawnKey:
    """Key for ``exp/trajectory_deviation/run_spawn_experiment.py`` state JSON."""

    cfg: str
    ep: str
    s: int
    n: int
    k_idx: int

    def encode(self) -> str:
        return (
            f"{self.cfg}:{self.ep}:"
            f"s{int(self.s)}:n{int(self.n)}:k{int(self.k_idx)}"
        )

    @classmethod
    def decode(cls, key: str) -> "SpawnKey":
        m = _SPAWN_KEY_RE.match(key)
        if not m:
            raise ValueError(f"Malformed SpawnKey: {key!r}")
        return cls(
            cfg=m.group("cfg"),
            ep=m.group("ep"),
            s=int(m.group("s")),
            n=int(m.group("n")),
            k_idx=int(m.group("k")),
        )
