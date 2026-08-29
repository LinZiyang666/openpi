"""Batch training loop for the X14 router (§3.6 / §3.7).

One iteration = one batch of N episodes = one Adam step = one new bundle
version. The loop rebuilds the TaskGraph and the ConductorDriver every round:
``ConductorDriver.__init__`` calls ``strategy.plan()`` once and exposes no
add-task API, so a driver's lifetime is exactly one round — which is also what
a repair round needs.

Per round::

    resolve resume state from the trainer checkpoint (the recovery authority)
      -> emit a versioned arm yaml pointing at THAT checkpoint's weights
      -> sample B-train inits -> dispatch -> barrier (client rows persisted per round)
      -> assemble package -> push to the server (marker last) -> remote three-source join
      -> complete?  no  -> repair round (new uid <orig>#r<n>), at most twice
                    yes -> remote trainer: one Adam step on the FULL N
      -> verify the exported checkpoint's meta version advanced -> reclaim shards
      -> next round's tasks carry the new bundle; workers rebind per task

**Init pool.** Training draws only from B-train, and the pool is explicit:
``--init-states-dir`` is REQUIRED and guarded. Left empty, LIBERO falls back to
the benchmark's own pruned_init — the official A pool — and the split's 0..49
indices would silently land on the one frozen test set.

Usage (conductor host)::

    PYTHONPATH=. uv run exp/rl_router/run_rl_router.py \\
        --matrix exp/rl_router/config/run_matrix.yaml --run-id l10_ts_lam1_s0 \\
        --arm-yaml exp/rl_router/config/libero_10/r_ts_train.yaml \\
        --init-states-dir exp/common/data/db_init/libero/libero_10 \\
        --artifacts exp/rl_router/data/artifacts.json \\
        --servers <host>:8000 --workers 8 --out-dir exp/rl_router/data/runs/l10_ts \\
        --remote-host <wls> --remote-port 14024 --remote-root /data/rl_router
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pathlib
import shlex
import threading
import time
from typing import Callable, Iterable, Optional

import numpy as np
import yaml

from openpi.cache.components.mlp_router_judge import RouterWeights
from openpi.cache.config import load_cache_config
from openpi.conductor import ConductorDriver, ServerEndpoint, WorkerAgent, WorkerSpec
from openpi.conductor import strategy as _strat
from openpi.conductor import task as _task

from exp.rl_router.batch_package import (
    LocalTransport,
    SshTransport,
    assemble_package,
    push_package,
    read_jsonl,
    remote_build_manifest,
    slot_of,
)

logger = logging.getLogger(__name__)

MAX_REPAIR_ROUNDS = 2

# M4 bootstrap size: the smoke batch plus the successor that proves the weight
# rollover reached the workers.
SMOKE_BATCHES = 2


# ---------------------------------------------------------------------------
# Init pool (B-train) — the guard that keeps the A pool untouched
# ---------------------------------------------------------------------------


def resolve_init_states_dir(path: str) -> str:
    """Validate the B-pool directory and return it, or fail loudly.

    Two failure modes, both silent without this check:

      - **empty path** — ``main._load_init_states`` falls back to
        ``task_suite.get_task_init_states()``, i.e. the official pruned_init A
        pool. The split's 0..49 indices would then address the frozen test set
        and every reported number would be contaminated;
      - **``.pruned_init`` shadowing** — the loader prefers ``<task>.pruned_init``
        over ``<task>.init`` inside a custom directory, so one stray file
        redirects that task back onto the test pool.

    The shadowing guard is the one the distillation collection already uses;
    reusing it keeps a single definition of "this directory is safe to train on".
    """
    if not path:
        raise SystemExit(
            "--init-states-dir is required: an empty value makes LIBERO load the "
            "official pruned_init pool (the frozen A-pool test set) and silently "
            "trains on it. Pass the diff pool, e.g. "
            "exp/common/data/db_init/libero/<suite>."
        )
    directory = pathlib.Path(path)
    if not directory.is_dir():
        raise SystemExit(f"--init-states-dir {path!r} is not a directory")

    from exp.ablation_study.build_distill_dataset import check_init_dir

    check_init_dir(str(directory))          # rejects any .pruned_init shadowing
    if not list(directory.glob("*.init")):
        raise SystemExit(f"--init-states-dir {path!r} holds no .init files")
    return str(directory)


# ---------------------------------------------------------------------------
# Init sampling
# ---------------------------------------------------------------------------


def btrain_pairs(split_path: str | pathlib.Path) -> list[tuple[int, int]]:
    """All ``(task_id, orig_init_state_idx)`` pairs in the B-train split.

    B-val is deliberately excluded and never sampled: it is the only held-out
    set the training loop can consult, and burning it in rollouts would leave
    nothing to select checkpoints on that is not also training data.
    """
    doc = yaml.safe_load(pathlib.Path(split_path).read_text(encoding="utf-8"))
    pairs: list[tuple[int, int]] = []
    for key, entry in sorted(doc.items()):
        task_id = int(key.rsplit("_", 1)[1])
        pairs.extend((task_id, int(idx)) for idx in entry["train"])
    return pairs


def sample_batch(
    pairs: list[tuple[int, int]], *, batch_size: int, batch_idx: int, seed: int
) -> list[tuple[int, int]]:
    """Deterministic without-replacement draw for round ``batch_idx``.

    Epochs are independent permutations of the whole pool, concatenated: every
    init is used once per epoch before any is reused, and a resumed run at the
    same ``batch_idx`` reproduces the same episodes. Reproducibility here is not
    a nicety — the interaction-efficiency curve's x-axis is a count of episodes,
    so a resume that re-drew different inits would silently redefine it.
    """
    if batch_size > len(pairs):
        raise ValueError(f"batch_size {batch_size} exceeds the B-train pool ({len(pairs)})")
    start = batch_idx * batch_size
    end = start + batch_size
    first_epoch, last_epoch = start // len(pairs), (end - 1) // len(pairs)
    ordered: list[tuple[int, int]] = []
    for epoch in range(first_epoch, last_epoch + 1):
        rng = np.random.RandomState(_derive_seed(seed, epoch))
        order = rng.permutation(len(pairs))
        ordered.extend(pairs[i] for i in order)
    offset = start - first_epoch * len(pairs)
    return ordered[offset:offset + batch_size]


def _derive_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}|{epoch}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Versioned arm yaml — how a new policy actually reaches the workers
# ---------------------------------------------------------------------------


def write_versioned_yaml(
    base_yaml: str | pathlib.Path,
    *,
    weights_path: str | pathlib.Path,
    weights_version: str,
    out_path: str | pathlib.Path,
    verify_meta: bool = True,
) -> pathlib.Path:
    """Emit the arm yaml for one weights version, atomically, and validate it.

    Without this the loop would ship the same static yaml every round and the
    workers would keep building a judge from the FIRST checkpoint while their
    task metadata advertised a new version — every episode would then be
    isolated on the version mismatch and no batch would ever fill.

    The checkpoint's own meta is checked against the version being advertised,
    so a mislabelled export cannot propagate into the fleet. ``verify_meta=False``
    is for a REMOTE ``weights_path``, which the conductor cannot open: that
    file's meta was verified on the host that wrote it (the trainer's
    ``export_meta.json``, checked by ``_verify_export``) or at publish time for
    the warm-start checkpoint.
    """
    if verify_meta:
        actual = RouterWeights.load(str(weights_path)).weights_version
        if actual != weights_version:
            raise ValueError(
                f"checkpoint {weights_path} carries weights_version {actual!r} but the "
                f"loop is advertising {weights_version!r}; refusing to ship a mislabelled policy"
            )
    cfg = copy.deepcopy(yaml.safe_load(pathlib.Path(base_yaml).read_text(encoding="utf-8")))
    cfg["checkpoints"]["cp1"]["judge"]["weights_path"] = str(weights_path)
    target = pathlib.Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    load_cache_config(str(tmp))             # allowlist + mlp_router rules, before dispatch
    tmp.replace(target)
    return target


# ---------------------------------------------------------------------------
# Resume — the trainer checkpoint is the authority
# ---------------------------------------------------------------------------


class RemoteRun:
    """The serving host's artifact namespace, addressed only through transport.

    The conductor (t107) and the server (wls) do not share a mount. Every
    trainer artifact — checkpoint, exported weights, metrics, and the shards
    themselves — lives on the server, so the loop must never ``open()`` those
    paths: it publishes and consumes them through the transport, and learns
    their state from small JSON summaries the trainer writes beside them.

    Local paths (journal, client rows, packages before push, run manifest) stay
    local. Keeping the two namespaces textually separate is what stops a remote
    path from being read as if it were local — the failure that made the first
    remote update the last one.
    """

    def __init__(self, transport, *, root: str, run_id: str, shard_root: str) -> None:
        self._transport = transport
        self.root = f"{root}/{run_id}"
        self.shard_root = f"{shard_root}/{run_id}"

    # -- addresses (remote strings, never local paths) --

    @property
    def checkpoint(self) -> str:
        return f"{self.root}/trainer_checkpoint.pt"

    @property
    def metrics(self) -> str:
        return f"{self.root}/metrics.jsonl"

    @property
    def state(self) -> str:
        return f"{self.root}/trainer_state.json"

    def weights(self, version: str) -> str:
        return f"{self.root}/weights/{version}.pt"

    def batch(self, batch_id: str) -> str:
        return f"{self.root}/{batch_id}"

    def package(self, batch_id: str, round_idx: int) -> str:
        # Round-scoped and immutable: a repair round's package differs from the
        # first round's by construction, so re-pushing into one directory would
        # trip the duplicate-digest guard that exists to catch two DIFFERENT
        # batches claiming one id. Separate generations keep both properties.
        return f"{self.batch(batch_id)}/package/r{round_idx}"

    def shards(self, batch_id: str) -> str:
        return f"{self.shard_root}/{batch_id}"

    # -- operations --

    def run(self, command: str) -> tuple[int, str]:
        return self._transport.run(command)

    def push_file(self, local: pathlib.Path, remote: str) -> None:
        """Publish a local file AT a remote address (destination name honoured)."""
        self._transport.push_file(local, remote)

    def fetch_json(self, remote: str, local: pathlib.Path) -> Optional[dict]:
        if not self._transport.fetch_file(remote, local):
            return None
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def exists(self, remote: str) -> bool:
        code, _ = self._transport.run(f"test -f {remote}")
        return code == 0


def refresh_remote_state(remote: RemoteRun, args) -> None:
    """Regenerate the remote state summary from the durable checkpoint.

    The state file is a cache of the checkpoint's ledger. Refreshing it before
    reading closes the window between "checkpoint written" and "state
    published": a crash in there would otherwise have the loop re-dispatch a
    batch the trainer had already consumed.
    """
    if not remote.exists(remote.checkpoint):
        return
    # PYTHONPATH must ride IN the command: the trainer-cmd template carries it,
    # but this state refresh builds its own command line — and an ssh remote
    # does not inherit the conductor's environment the way LocalTransport
    # does. Single-host resumes worked by that inheritance accident; the first
    # cross-host resume died here with "No module named 'exp'" (2026-08-19).
    code, output = remote.run(
        f"cd {args.remote_workdir} && PYTHONPATH=. {args.remote_python} exp/rl_router/train_router.py --state-only "
        f"--checkpoint {remote.checkpoint} --state-out {remote.state} "
        f"--manifest /dev/null --package /dev/null --shards /dev/null "
        f"--weights-out /dev/null --metrics /dev/null --lam 0 --t-max 1 --arm-costs {{}}"
    )
    if code != 0:
        raise SystemExit(f"ALERT: could not refresh remote trainer state: {output.strip()[-1000:]}")


def resume_state(remote: RemoteRun, scratch: pathlib.Path) -> dict:
    """Derive where to restart from the remote trainer state, not from CLI flags.

    A ``--start-batch`` supplied by an operator is a guess; the trainer's state
    summary records exactly which batches were consumed and which version it
    holds. Reading it (rather than a local file) is what makes resume correct on
    the real two-host topology.

    A checkpoint whose exported weights are missing is the save/export crash
    window, not a corrupt run: the update is already durable inside the
    checkpoint, so the caller replays just the export.
    """
    state = remote.fetch_json(remote.state, scratch / "trainer_state.json")
    if state is None:
        return {"next_batch_idx": 0, "weights_version": None, "consumed": [],
                "needs_export": False, "missing_metrics": []}
    version = str(state["weights_version"])
    consumed = list(state.get("consumed_batches", []))
    indices = [int(b["batch_id"][1:]) for b in consumed if b.get("batch_id", "").startswith("b")]
    # Reconcile the ledger against metrics INDEPENDENTLY of the weights. The
    # crash windows are separate: "export never happened" leaves the weights
    # missing, but "export happened, metrics did not" leaves them present. Keying
    # recovery off the weights alone silently loses that batch's row forever.
    metrics_rows = []
    if remote._transport.fetch_file(remote.metrics, scratch / "metrics.jsonl"):
        metrics_rows = read_jsonl(scratch / "metrics.jsonl")
    recorded = {row.get("batch_id") for row in metrics_rows if not row.get("admission_failed")}
    missing_metrics = [
        b["batch_id"] for b in consumed
        if b.get("batch_id") and b["batch_id"] not in recorded
    ]
    return {
        "next_batch_idx": (max(indices) + 1) if indices else 0,
        "weights_version": version,
        "consumed": consumed,
        "needs_export": not remote.exists(remote.weights(version)),
        "missing_metrics": missing_metrics,
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class RouterBatchStrategy(_strat.ExperimentStrategy):
    """One eval stage carrying exactly this round's episodes.

    Every episode's ``extra`` carries the identity the server needs to name its
    feature shard (``run_id`` / ``batch_id`` / ``weights_version``);
    ``task_uid`` and ``attempt`` are NOT stamped here — the runner takes them
    from the dispatched task, because ``extra`` does not follow a requeue.
    """

    def __init__(
        self,
        *,
        suite: str,
        yaml_path: str,
        run_id: str,
        batch_id: str,
        weights_version: str,
        bundle_id: str,
        slots: list[tuple[int, int, str]],
        trials_per_task: int,
    ) -> None:
        self._suite = suite
        self._yaml_path = yaml_path
        self._run_id = run_id
        self._batch_id = batch_id
        self._weights_version = weights_version
        self._bundle_id = bundle_id
        self._slots = slots            # (task_id, orig_init_idx, task_uid)
        self._trials = trials_per_task

    def plan(self, yamls, server_assignment) -> _task.TaskGraph:
        graph = _task.TaskGraph()
        yaml_id = yamls[0]
        server = server_assignment[yaml_id]
        stage = _task.Stage(
            stage_id=f"{yaml_id}:{self._batch_id}", yaml_id=yaml_id, phase="eval",
            server=server, setup={"yaml_path": self._yaml_path},
        )
        for episode_idx, (task_id, init_idx, task_uid) in enumerate(self._slots):
            stage.episodes.append(_task.EpisodeTask(
                task_uid=task_uid, yaml_id=yaml_id, phase="eval", experiment=self._suite,
                task_id=task_id, episode_idx=episode_idx, orig_init_state_idx=init_idx,
                server_host=server.host, server_port=server.port,
                bundle_id=self._bundle_id,
                extra={
                    "num_trials_per_task": self._trials,
                    "run_id": self._run_id,
                    "batch_id": self._batch_id,
                    "weights_version": self._weights_version,
                },
            ))
        graph.add_stage(stage)
        return graph

    def on_stage_begin(self, stage, ctl, ctx) -> None:
        # Version-scoped bundle id: the workers rebind per task, so a batch
        # barrier is the only moment a new policy can reach them.
        ctl.load_cache_config(
            yaml_content=pathlib.Path(stage.setup["yaml_path"]).read_text(encoding="utf-8"),
            yaml_id=f"{stage.yaml_id}__{self._weights_version}",
            bundle_id=self._bundle_id,
        )

    def on_stage_complete(self, stage, ctl, ctx) -> None:
        pass


def make_slots(
    pairs: Iterable[tuple[int, int]], *, yaml_id: str, repair: int = 0
) -> list[tuple[int, int, str]]:
    """Attach a task_uid to each (task, init) pair; repairs get ``#r<n>``."""
    slots = []
    for episode_idx, (task_id, init_idx) in enumerate(pairs):
        uid = _task.make_task_uid(yaml_id, "eval", task_id, episode_idx)
        slots.append((task_id, init_idx, f"{uid}#r{repair}" if repair else uid))
    return slots


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------


class _ResultRowPersister:
    """Append each episode's client rows the moment its result arrives.

    ``ConductorDriver`` offers two row sinks and only one of them is prompt:
    ``per_step_writer`` fires at *stage completion*, while ``monitor.on_result``
    fires per episode inside ``handle_result`` — right where the journal record
    is written. That difference decides whether a crash is survivable: the
    journal makes a restarted driver skip already-completed uids, so any row
    still buffered in memory when the process dies can never be re-collected,
    and the three-source join for those episodes is permanently impossible. The
    repair budget would then be spent re-running episodes whose evidence simply
    vanished.

    Rows are deduplicated on ``(task_uid, attempt, decision_idx)`` so the
    stage-completion drain and the run() finalizer cannot double-append what
    this already wrote.
    """

    def __init__(self, rows_path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(rows_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: set[tuple] = set()
        for row in read_jsonl(self._path):          # resume: do not re-add
            self._seen.add(self._key(row))

    @staticmethod
    def _key(row: dict) -> tuple:
        ro = row.get("router_outputs") or {}
        return (str(row.get("task_uid")), int(row.get("attempt", -1)),
                row.get("_kind"), ro.get("decision_idx"), row.get("step_idx"))

    def write(self, rows: list[dict]) -> int:
        written = 0
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            for row in rows:
                key = self._key(row)
                if key in self._seen:
                    continue
                self._seen.add(key)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
            os.fsync(fh.fileno())
        return written

    # -- driver seams --

    def on_result(self, result) -> None:
        """Driver monitor hook: fires per episode, alongside the journal write."""
        rows = list(getattr(result, "per_step_rows", None) or [])
        for row in rows:
            row.setdefault("task_uid", result.task_uid)
            row.setdefault("attempt", result.attempt)
        if rows:
            self.write(rows)

    def on_progress(self, payload) -> None:
        """Monitor protocol completeness; progress carries no rows."""

    def per_step_writer(self, _yaml_id: str, rows: list[dict]) -> None:
        self.write(rows)


def run_round(
    *,
    strategy: RouterBatchStrategy,
    yaml_id: str,
    servers: list[ServerEndpoint],
    worker_specs: list[WorkerSpec],
    journal_path: str,
    rows_path: str,
    bind_host: str,
    episode_timeout_s: float,
) -> tuple[list[dict], list[dict]]:
    """Dispatch one round to completion. Returns ``(journal_rows, client_rows)``."""
    persister = _ResultRowPersister(rows_path)
    driver = ConductorDriver(
        strategy,
        yaml_weights={yaml_id: 100},
        servers=servers,
        journal_path=journal_path,
        ctl_factory=_default_ctl_factory(),
        episode_timeout_s=episode_timeout_s,
        bind_host=bind_host,
        per_step_writer=persister.per_step_writer,
        monitor=persister,
    )
    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    agent = WorkerAgent(worker_specs, driver_host=bind_host, driver_port=driver.port)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    driver_thread.join()
    agent.stop()
    agent_thread.join(timeout=30)

    # Belt and braces: anything the per-result hook somehow missed. Deduped by
    # the persister, so this cannot double-append.
    leftovers = list(driver.per_step_rows)
    if leftovers:
        persister.write(leftovers)
    return read_jsonl(journal_path), read_jsonl(rows_path)


def _default_ctl_factory():
    from examples.libero.episode_runner import default_client_factory

    return default_client_factory


# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------


def run_batch_with_repair(
    *,
    round_runner: Callable[[RouterBatchStrategy, int], tuple[list[dict], list[dict]]],
    strategy_factory: Callable[[list[tuple[int, int, str]], int], RouterBatchStrategy],
    slots: list[tuple[int, int, str]],
    join: Callable[[list[dict], list[dict], int, list[tuple[str, int]]], tuple[bool, list[str]]],
    train: Callable[[int], tuple[bool, list[dict]]],
    batch_id: str,
    state_path: Optional[str | pathlib.Path] = None,
) -> int:
    """Run a batch to a landed update, repairing at most ``MAX_REPAIR_ROUNDS`` times.

    A batch can fail to fill in **two** ways, and both feed the same bounded
    state machine:

      1. **structural** — a slot has no admissible episode (shard missing,
         stale attempt, errored, discontinuous);
      2. **parity** — the batch filled, but the trainer's bitwise on-policy
         check rejected an episode. Its ``(uid, attempt)`` is quarantined so the
         manifest's deterministic priority stops re-selecting the same bad
         attempt, and its slot is re-run.

    Treating only the first as repairable was the earlier gap: a parity failure
    would abort the run instead of being repaired, even though it is exactly the
    situation §3.5's "repair, then update on the full N" describes.

    Repairs re-run the *init* under a new ``#r<n>`` uid, so a failed attempt and
    its replacement never contend for one shard path. Returns the round index
    whose update landed.
    """
    # The repair generation is DURABLE. Journal and client rows are cumulative
    # and append-only, so a process that restarts at round 0 would rebuild an
    # "r0" package containing round-1 rows — a different digest for a directory
    # the server already holds, which the duplicate-batch guard correctly
    # refuses. Resuming at the round that was actually in flight keeps each
    # generation's immutable package consistent with its own content.
    state = _load_repair_state(state_path)
    pending = _pending_for(slots, state)
    missing: list[str] = list(state.get("missing", []))
    quarantine: list[tuple[str, int]] = [tuple(q) for q in state.get("quarantine", [])]
    first_round = int(state.get("round", 0))
    if first_round:
        logger.info("batch %s: resuming at repair round %d (%d slot(s) pending)",
                    batch_id, first_round, len(pending))

    for attempt_round in range(first_round, MAX_REPAIR_ROUNDS + 1):
        strategy = strategy_factory(pending, attempt_round)
        journal_rows, client_rows = round_runner(strategy, attempt_round)
        complete, missing = join(journal_rows, client_rows, attempt_round, quarantine)

        if complete:
            landed, rejected = train(attempt_round)
            if landed:
                return attempt_round
            quarantine.extend(
                (str(r["task_uid"]), int(r["attempt"])) for r in rejected
            )
            missing = sorted({slot_of(str(r["task_uid"])) for r in rejected})
            logger.warning(
                "batch %s round %d: trainer rejected %d episode(s) on parity -> repair",
                batch_id, attempt_round, len(rejected),
            )
        else:
            logger.warning(
                "batch %s round %d: %d slot(s) missing -> repair", batch_id,
                attempt_round, len(missing),
            )

        wanted = set(missing)
        pending = [
            (task_id, init_idx, f"{slot_of(uid)}#r{attempt_round + 1}")
            for task_id, init_idx, uid in slots if slot_of(uid) in wanted
        ]
        _save_repair_state(state_path, round_idx=attempt_round + 1,
                           missing=missing, quarantine=quarantine)
        if not pending:
            raise RuntimeError(
                f"ALERT: batch {batch_id} needs repair but no slot could be identified "
                f"(missing={missing}); halting for owner adjudication"
            )

    raise RuntimeError(
        f"ALERT: batch {batch_id} still short {len(missing)} slot(s) after "
        f"{MAX_REPAIR_ROUNDS} repair rounds; halting for owner adjudication rather "
        "than updating on a shrunken batch"
    )


def _load_repair_state(path: Optional[str | pathlib.Path]) -> dict:
    if not path or not pathlib.Path(path).exists():
        return {}
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_repair_state(path: Optional[str | pathlib.Path], *, round_idx: int,
                       missing: list[str], quarantine: list[tuple[str, int]]) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps({
        "round": round_idx, "missing": list(missing),
        "quarantine": [[u, a] for u, a in quarantine],
    }, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def _pending_for(slots: list[tuple[int, int, str]], state: dict) -> list[tuple[int, int, str]]:
    """Slots to dispatch for the persisted generation (all of them at round 0)."""
    round_idx = int(state.get("round", 0))
    if not round_idx:
        return slots
    wanted = set(state.get("missing", []))
    return [
        (task_id, init_idx, f"{slot_of(uid)}#r{round_idx}")
        for task_id, init_idx, uid in slots if slot_of(uid) in wanted
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def worker_gpu_id(i: int, *, gpus: int, gpu_ids: str = "") -> str:
    """CUDA device for worker ``i``.

    ``--gpus N`` assumes the healthy devices are exactly ``0..N-1`` — true on a
    box we own, false on a shared cluster where the free VRAM lives on whatever
    cards the other users happen not to fill (a worker landing on a full GPU
    dies at EGL init, silently costing its whole slot share). ``--gpu-ids``
    names the devices explicitly and is cycled, so repetition weights a card:
    ``7,7,7,4`` puts three quarters of the fleet on device 7.
    """
    ids = [g.strip() for g in gpu_ids.split(",") if g.strip()] if gpu_ids else []
    if ids:
        return ids[i % len(ids)]
    return str(i % gpus)


def _resolve_lambda_or_placeholder(matrix: dict, run: dict, *, bootstrap: bool) -> float:
    """Resolve λ, or hand the bootstrap a declared placeholder.

    A formal run must never substitute a default: it would report results
    against the pre-registered λ₁/λ₂ labels while λ came from nowhere. The M4
    bootstrap is the one exception the plan names (§1, "placeholder constants
    only here"), because it runs BEFORE the pilot that calibrates λ. The
    placeholder is the middle of the frozen grid rather than zero, so the cost
    term is actually exercised instead of being silently switched off.
    """
    from exp.rl_router.launch_gates import resolve_lambda
    from exp.rl_router.pilot_lambda import LAMBDA_GRID

    try:
        return resolve_lambda(matrix, run)
    except SystemExit:
        if not bootstrap:
            raise
        placeholder = float(sorted(LAMBDA_GRID)[len(LAMBDA_GRID) // 2])
        logger.warning(
            "M4 bootstrap: lambda %s is not calibrated yet; using the frozen grid's "
            "midpoint %.3f as a PLACEHOLDER. This run's products are barred from "
            "A-pool evaluation and the paper (plan §1).", run["lambda"], placeholder,
        )
        return placeholder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="config/run_matrix.yaml")
    parser.add_argument("--run-id", required=True, help="a run id from the matrix")
    parser.add_argument("--arm-yaml", required=True, help="base arm yaml (weights_path is rewritten)")
    parser.add_argument("--init-states-dir", required=True, help="B-train diff pool")
    parser.add_argument("--artifacts", required=True, help="json of M5a/M5b/M5c/M4 artifact paths")
    parser.add_argument("--servers", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", required=True, help="LOCAL conductor-side outputs")
    parser.add_argument("--shard-root", required=True, help="REMOTE server-side dump root")
    parser.add_argument("--remote-root", required=True, help="REMOTE artifact root")
    parser.add_argument("--remote-host", default="", help="empty = same-host (LocalTransport)")
    parser.add_argument("--remote-port", type=int, default=14024)
    parser.add_argument("--remote-user", default="")
    parser.add_argument("--remote-workdir", default=".", help="repo path on the server")
    parser.add_argument("--remote-python", default="uv run",
                        help="how to launch python ON THE SERVER. A non-interactive ssh "
                             "session gets the system PATH, which on a real deployment does "
                             "not contain a user-local uv, so this must be settable — "
                             "e.g. '/home/<user>/.local/bin/uv run'")
    parser.add_argument("--trainer-cmd", required=True,
                        help="remote command template; receives {manifest} {package} "
                             "{shards} {weights_in} {weights_out} {checkpoint} {metrics} "
                             "{export_meta} {state} {rejected} {lam} {t_max} {arm_costs}")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu-ids", default="",
                        help="explicit CUDA device list, cycled across workers "
                             "(e.g. '7,7,7,4'); overrides --gpus — see worker_gpu_id")
    parser.add_argument("--conda-env", default="")
    parser.add_argument("--episode-timeout-s", type=float, default=1800.0)
    parser.add_argument("--smoke", action="store_true",
                        help="M4 bootstrap: run SMOKE_EPISODES x 2 batches and emit the "
                             "capacity report the formal launch gate requires")
    parser.add_argument("--pilot", action="store_true",
                        help="this run IS one λ-pilot candidate (M5c). Waives only the "
                             "pilot-record gate — the record is what this run helps "
                             "produce — and keeps every other launch gate, λ included: "
                             "a candidate is defined by the λ it was told to train")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from exp.rl_router.launch_gates import (
        build_run_manifest,
        check_launch_gates,
        load_run_matrix,
        resolve_run,
    )

    matrix = load_run_matrix(args.matrix)
    run = resolve_run(matrix, args.run_id)
    artifacts = json.loads(pathlib.Path(args.artifacts).read_text(encoding="utf-8"))
    init_dir = resolve_init_states_dir(args.init_states_dir)

    from exp.rl_router.launch_gates import SMOKE_EPISODES

    suite = run["suite"]
    if args.smoke:
        # The bootstrap that breaks the "need a report to run, need a run to
        # report" circle. Its size is MECHANICALLY fixed, never derived from the
        # matrix: a --smoke that silently ran the formal 100 x 40 would spend the
        # entire run budget and still produce a report the M4 gate rejects for
        # covering the wrong episode count. The successor batch is what proves
        # the weight rollover reached the workers, hence exactly two.
        batch_size, total_batches = SMOKE_EPISODES, SMOKE_BATCHES
    else:
        batch_size = int(matrix["batch_size"])
        # An amendment row may carry its own episode budget; the launch gate
        # honours it only when the row declares the amendment, so this stays
        # exactly what the gate approved.
        total_batches = int(run.get("episodes", matrix["episodes_per_run"])) // batch_size
    transport_probe = (
        LocalTransport() if not args.remote_host
        else SshTransport(args.remote_host, port=args.remote_port, user=args.remote_user)
    )
    problems = check_launch_gates(
        matrix, run, artifacts=artifacts,
        # The gate checks the run's CONFIGURATION against the frozen decisions;
        # the bootstrap does not change them, it just executes a smaller
        # diagnostic, so the frozen batch size is what is declared here.
        batch_size=int(matrix["batch_size"]), seed=int(run["seed"]),
        variant=run["variant"], suite=suite,
        # Capacity is checked WHERE THE SHARDS ARE; a local stat of a remote
        # path would silently report zero.
        remote_live_bytes=_remote_live_bytes(transport_probe, args),
        arm_yaml=args.arm_yaml,
        planned_batches=None if args.smoke else total_batches,
        bootstrap=args.smoke,
        pilot_calibration=args.pilot,
    )
    if problems:
        raise SystemExit("LAUNCH BLOCKED:\n" + "\n".join(f"  - {p}" for p in problems))
    lam = _resolve_lambda_or_placeholder(matrix, run, bootstrap=args.smoke)

    out_dir = pathlib.Path(args.out_dir)
    scratch = out_dir / "_fetched"
    scratch.mkdir(parents=True, exist_ok=True)
    remote = RemoteRun(transport_probe, root=args.remote_root, run_id=args.run_id,
                       shard_root=args.shard_root)

    arm_costs = json.loads(
        pathlib.Path(artifacts["arm_costs"]).read_text(encoding="utf-8")
    )["normalized_costs"]
    pairs = btrain_pairs(matrix["suites"][suite]["split"])
    # Charge the interaction ledger by the candidates the pilot ACTUALLY ran: a
    # supplementary λ is a real cost, and defaulting to the grid size would
    # under-report exactly the runs that needed extra calibration. The bootstrap
    # precedes the pilot, so there is nothing to charge yet and the ledger falls
    # back to the grid size — its manifest is barred from the paper anyway (§1).
    pilot_path = pathlib.Path(artifacts.get("pilot", ""))
    pilot_record = (
        json.loads(pilot_path.read_text(encoding="utf-8"))
        if artifacts.get("pilot") and pilot_path.exists() else {}
    )
    manifest_doc = build_run_manifest(
        matrix=matrix, run=run, lam=lam, init_states_dir=init_dir, arm_costs=arm_costs,
        warmstart_episodes=len(pairs), artifacts=artifacts,
        pilot_candidates=pilot_record.get("candidates_run"),
        # Identity a pilot candidate must be able to prove it ran under.
        split_path=matrix["suites"][suite]["split"], arm_yaml=args.arm_yaml,
        judge_mode=str((yaml.safe_load(pathlib.Path(args.arm_yaml).read_text(
            encoding="utf-8"))["checkpoints"]["cp1"]["judge"]).get("mode", "")),
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest_doc, indent=2), encoding="utf-8"
    )
    logger.info("run manifest written; interaction ledger offset = %d episodes",
                manifest_doc["interaction_ledger"]["shared_offset"])

    # ---- resume from the REMOTE trainer state -------------------------------
    refresh_remote_state(remote, args)          # checkpoint is the authority
    state = resume_state(remote, scratch)
    if state["weights_version"] is None:
        # First launch: publish the warm-start checkpoint to the serving host.
        # Verify locally first — this is the one moment the file is local.
        local_warm = pathlib.Path(artifacts["warmstart_weights"])
        version = RouterWeights.load(str(local_warm)).weights_version
        remote.push_file(local_warm, remote.weights(version))
        logger.info("published warm-start weights %s -> %s", version, remote.weights(version))
        weights_version, start_batch = version, 0
    else:
        weights_version, start_batch = state["weights_version"], state["next_batch_idx"]
        if state["needs_export"] or state["missing_metrics"]:
            # Two independent tail windows, one recovery: the export replays
            # idempotently and the same pass reconciles every consumed batch's
            # metrics row against the ledger. Gating this on the weights alone
            # was the bug — a crash AFTER the export leaves them present.
            logger.warning(
                "resume: recovering the update tail (needs_export=%s, metrics missing for %s)",
                state["needs_export"], state["missing_metrics"] or "none",
            )
            _remote_reexport(remote, args, weights_version)
        # A crash after the update but before reclaim leaves the batch's shards
        # behind. The ledger gate makes reclaim safe and idempotent, so sweep
        # every consumed batch rather than carrying ~2.6 GB per orphan forward.
        _reclaim_consumed(remote, args, state["consumed"])
        logger.info("resuming: %s, next batch %d", weights_version, start_batch)

    servers = [ServerEndpoint(*_split_endpoint(s)) for s in args.servers.split(",")]
    yaml_id = pathlib.Path(args.arm_yaml).stem
    specs = [
        WorkerSpec(
            worker_id=f"w{i}", server_key=servers[i % len(servers)].key,
            gpu_id=worker_gpu_id(i, gpus=args.gpus, gpu_ids=args.gpu_ids),
            conda_env=args.conda_env,
            task_suite_name=suite,
            # THE B-pool binding. Empty here = the official pruned_init A pool.
            init_states_dir=init_dir,
        )
        for i in range(args.workers)
    ]

    for batch_idx in range(start_batch, total_batches):
        batch_id = f"b{batch_idx:04d}"
        weights_version = run_one_batch(
            batch_idx=batch_idx, batch_id=batch_id, weights_version=weights_version,
            remote=remote, args=args, run=run, matrix=matrix, pairs=pairs,
            servers=servers, specs=specs, yaml_id=yaml_id, out_dir=out_dir,
            scratch=scratch, lam=lam, arm_costs=arm_costs, batch_size=batch_size,
        )

    if args.smoke:
        report = emit_m4_report(remote=remote, args=args, out_dir=out_dir,
                                scratch=scratch, episodes=batch_size)
        path = out_dir / "m4_smoke.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(f"M4 SMOKE FAILED: {report['violations']}")
        logger.info("M4 report -> %s (feed it to launch_gates as capacity_smoke)", path)


def run_one_batch(
    *, batch_idx: int, batch_id: str, weights_version: str, remote: RemoteRun, args,
    run: dict, matrix: dict, pairs: list, servers, specs, yaml_id: str,
    out_dir: pathlib.Path, scratch: pathlib.Path, lam: float, arm_costs: dict,
    batch_size: int,
) -> str:
    """Run one batch end to end and return the weights version it produced."""
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    bundle_id = f"rlr_{args.run_id}_{weights_version}"

    # The yaml is read by the SERVER, so its weights_path must be the remote
    # address; the meta check happened when that file was published.
    versioned_yaml = write_versioned_yaml(
        args.arm_yaml, weights_path=remote.weights(weights_version),
        weights_version=weights_version, out_path=batch_dir / f"{yaml_id}__{weights_version}.yaml",
        verify_meta=False,
    )
    slots = make_slots(
        sample_batch(pairs, batch_size=batch_size, batch_idx=batch_idx, seed=int(run["seed"])),
        yaml_id=yaml_id,
    )
    expected = [slot_of(uid) for _, _, uid in slots]
    remote_shards = remote.shards(batch_id)
    remote_manifest = f"{remote.batch(batch_id)}/manifest.json"
    next_version = _next_version(weights_version)
    state: dict = {}

    def factory(pending, repair):
        return RouterBatchStrategy(
            suite=run["suite"], yaml_path=str(versioned_yaml), run_id=args.run_id,
            batch_id=batch_id, weights_version=weights_version, bundle_id=bundle_id,
            slots=pending, trials_per_task=batch_size,
        )

    def runner(strategy, repair):
        return run_round(
            strategy=strategy, yaml_id=yaml_id, servers=servers, worker_specs=specs,
            journal_path=str(batch_dir / "journal.jsonl"),
            rows_path=str(batch_dir / "client_rows.jsonl"),
            bind_host=args.bind_host, episode_timeout_s=args.episode_timeout_s,
        )

    def join(journal_rows, client_rows, round_idx, quarantine):
        """Package -> push (round-scoped) -> remote three-source join -> missing."""
        local_pkg = assemble_package(
            batch_dir / "package" / f"r{round_idx}", batch_id=batch_id,
            weights_version=weights_version, journal_rows=journal_rows,
            client_rows=client_rows, expected_slots=expected, quarantine=quarantine,
        )
        remote_pkg = remote.package(batch_id, round_idx)
        push_package(remote._transport, local_pkg, remote_pkg)
        ok, output = remote_build_manifest(
            remote._transport, remote_package=remote_pkg, remote_shards=remote_shards,
            remote_manifest=remote_manifest, workdir=args.remote_workdir,
            python=args.remote_python,
        )
        doc = remote.fetch_json(remote_manifest, batch_dir / "remote_manifest.json")
        if doc is None:
            logger.warning("remote join produced no manifest: %s", output.strip()[-500:])
            return False, expected
        return bool(doc.get("complete")) and ok, list(doc.get("missing_slots", []))

    def train(round_idx):
        """Remote Adam step. Returns ``(landed, rejected_episodes)``."""
        remote_pkg = remote.package(batch_id, round_idx)
        code, output = remote.run(
            f"cd {args.remote_workdir} && " + args.trainer_cmd.format(
                manifest=remote_manifest, package=remote_pkg, shards=remote_shards,
                weights_in=remote.weights(weights_version),
                weights_out=remote.weights(next_version), checkpoint=remote.checkpoint,
                metrics=remote.metrics,
                export_meta=f"{remote.batch(batch_id)}/export_meta.json",
                state=remote.state, rejected=f"{remote.batch(batch_id)}/rejected.json",
                lam=lam, t_max=matrix.get("t_max", 520),
                arm_costs=shlex.quote(json.dumps(arm_costs)),
            )
        )
        if code == 0:
            state.update(_verify_export(remote, batch_id, batch_dir, next_version))
            return True, []
        rejected = remote.fetch_json(
            f"{remote.batch(batch_id)}/rejected.json", batch_dir / "rejected.json"
        )
        if rejected is None:
            raise RuntimeError(
                f"ALERT: trainer exited {code} without an admission report: "
                f"{output.strip()[-2000:]}"
            )
        return False, list(rejected.get("rejected", []))

    landed_round = run_batch_with_repair(
        round_runner=runner, strategy_factory=factory, slots=slots,
        join=join, train=train, batch_id=batch_id,
        state_path=batch_dir / "repair_state.json",
    )

    (batch_dir / "versions.json").write_text(
        json.dumps([weights_version, state["weights_version"]]), encoding="utf-8"
    )
    # Reclaim WHERE THE SHARDS ARE, gated on the ledger naming this batch.
    package_meta = json.loads(
        (batch_dir / "package" / f"r{landed_round}" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    before = _remote_live_bytes(remote._transport, args)
    code, output = remote.run(
        f"cd {args.remote_workdir} && {args.remote_python} exp/rl_router/batch_package.py reclaim "
        f"--shards {remote_shards} --checkpoint {remote.checkpoint} "
        f"--batch-id {batch_id} --package-sha256 {package_meta['package_sha256']}"
    )
    if code != 0:
        raise RuntimeError(f"ALERT: remote reclaim failed for {batch_id}: {output.strip()[-1000:]}")
    after = _remote_live_bytes(remote._transport, args)
    # Peak-vs-steady evidence, measured rather than estimated: the capacity gate
    # is about the high-water mark, which is only visible either side of reclaim.
    (batch_dir / "capacity.json").write_text(
        json.dumps({"before": before, "after": after}), encoding="utf-8"
    )
    logger.info("batch %s done: %s -> %s (round %d)", batch_id, weights_version,
                state["weights_version"], landed_round)
    return state["weights_version"]


def emit_m4_report(*, remote: RemoteRun, args, out_dir: pathlib.Path,
                   scratch: pathlib.Path, episodes: int) -> dict:
    """Build the M4 report from the run's OWN artifacts, bound to their identity.

    Everything it asserts is fetched from what the batches actually produced —
    the remote join, the remote metrics, the successor batch's shard manifest,
    and the byte counts measured either side of reclaim. A report assembled from
    anything else is a claim about a different run.
    """
    from exp.rl_router.batch_package import load_shard_manifest
    from exp.rl_router.launch_gates import m4_smoke

    first, second = out_dir / "b0000", out_dir / "b0001"
    manifest = json.loads((first / "remote_manifest.json").read_text(encoding="utf-8"))
    versions = json.loads((first / "versions.json").read_text(encoding="utf-8"))
    metrics_local = scratch / "metrics.jsonl"
    remote._transport.fetch_file(remote.metrics, metrics_local)
    metrics = [row for row in read_jsonl(metrics_local)
               if row.get("batch_id") == "b0000"]
    capacity = json.loads((first / "capacity.json").read_text(encoding="utf-8"))
    next_shards_local = scratch / "next_manifest.jsonl"
    remote._transport.fetch_file(f"{remote.shards('b0001')}/manifest.jsonl", next_shards_local)
    next_shards = read_jsonl(next_shards_local)
    package_meta = json.loads(
        (first / "package" / "r0" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    del second, load_shard_manifest
    return m4_smoke(
        manifest=manifest, metrics=metrics, checkpoint_versions=versions,
        next_batch_shards=next_shards, dump_root=scratch, episodes=episodes,
        run_id=args.run_id, batch_id="b0000",
        package_sha256=package_meta["package_sha256"],
        bytes_before_reclaim=capacity["before"],
        bytes_after_reclaim=capacity["after"],
        # Measured on the serving host either side of reclaim; `scratch` is the
        # conductor's local fetch dir and stat-ing it would report ~0 bytes of
        # steady state for a dump root it cannot even see.
        live_bytes=capacity["after"],
    )


def _verify_export(remote: RemoteRun, batch_id: str, batch_dir: pathlib.Path,
                   expected_version: str) -> dict:
    """Confirm the remote export landed and advanced by exactly one version.

    Read from the summary the trainer wrote on the machine that wrote the
    weights — the conductor cannot stat that file, and "the version changed" is
    not the contract: skipping a version means an update was lost.
    """
    meta = remote.fetch_json(
        f"{remote.batch(batch_id)}/export_meta.json", batch_dir / "export_meta.json"
    )
    if meta is None:
        raise RuntimeError(f"ALERT: trainer produced no export meta for {batch_id}")
    if meta["weights_version"] != expected_version:
        raise RuntimeError(
            f"ALERT: trainer exported {meta['weights_version']!r}, expected "
            f"{expected_version!r} (exactly one increment)"
        )
    return meta


def _remote_reexport(remote: RemoteRun, args, version: str) -> None:
    code, output = remote.run(
        f"cd {args.remote_workdir} && {args.remote_python} exp/rl_router/train_router.py --export-only "
        f"--checkpoint {remote.checkpoint} --weights-out {remote.weights(version)} "
        f"--state-out {remote.state} --manifest /dev/null --package /dev/null "
        f"--shards /dev/null --metrics {remote.metrics} --lam 0 --t-max 1 --arm-costs {{}}"
    )
    if code != 0:
        raise SystemExit(f"ALERT: re-export failed: {output.strip()[-1000:]}")


def _reclaim_consumed(remote: RemoteRun, args, consumed: list[dict]) -> None:
    """Sweep shards of batches the ledger already consumed (idempotent).

    Only batches whose shard directory still EXISTS are swept, decided by one
    remote listing rather than one remote call per consumed batch. Reclaim was
    always idempotent, so re-running it on an already-swept batch was correct —
    just ruinously expensive: every call is a fresh ssh plus a `uv run` cold
    start, measured at 2.7 s per batch same-host and 4.0 s through the relay.
    That is linear in progress, so a resume cost ~15 min at batch 227 and would
    have cost ~40 min at batch 600, every time — for work that is a no-op in all
    but the crash window this sweep exists to cover.

    The invariant is unchanged: a batch whose shards survived (crash after the
    trainer update, before the sweep) is still reclaimed. A batch with no shard
    directory has nothing to reclaim by definition.
    """
    batch_ids = [e["batch_id"] for e in consumed if e.get("batch_id")]
    if not batch_ids:
        return
    # Fails OPEN, deliberately: an unlistable shard root means "I cannot tell
    # what survived", and the safe answer there is to sweep everything rather
    # than silently carry ~2.6 GB per orphan forward.
    code, output = remote.run(f"ls -1 {remote.shard_root} 2>/dev/null")
    if code == 0:
        present = {line.strip() for line in output.splitlines() if line.strip()}
        stale = [b for b in batch_ids if b in present]
    else:
        logger.warning("resume: could not list %s; sweeping every consumed batch",
                       remote.shard_root)
        stale = batch_ids
    if not stale:
        logger.info("resume: no leftover shards among %d consumed batches", len(batch_ids))
        return
    logger.info("resume: reclaiming %d leftover shard dir(s) of %d consumed batches",
                len(stale), len(batch_ids))
    for batch_id in stale:
        code, output = remote.run(
            f"cd {args.remote_workdir} && {args.remote_python} exp/rl_router/batch_package.py reclaim "
            f"--shards {remote.shards(batch_id)} --checkpoint {remote.checkpoint} "
            f"--batch-id {batch_id}"
        )
        if code != 0:
            logger.warning("resume reclaim for %s reported: %s", batch_id,
                           output.strip()[-300:])


def _remote_live_bytes(transport, args) -> int:
    """Live feature bytes under the REMOTE dump root. Fails closed.

    An unmeasurable disk is not an empty disk. Returning "unknown" and letting
    the gate shrug would clear the capacity check precisely when the serving
    host is unreachable or its layout has changed — the two situations where a
    4k-episode run is most likely to fill it.
    """
    code, output = transport.run(
        f"cd {args.remote_workdir} && {args.remote_python} exp/rl_router/batch_package.py capacity "
        f"--root {args.shard_root}"
    )
    if code != 0:
        raise SystemExit(
            f"LAUNCH BLOCKED: could not measure remote capacity under {args.shard_root} "
            f"(exit {code}): {output.strip()[-500:]}"
        )
    live = _last_json_field(output, "live_bytes")
    if live is None:
        raise SystemExit(
            f"LAUNCH BLOCKED: remote capacity probe returned unparsable output: "
            f"{output.strip()[-500:]}"
        )
    return live


def _last_json_field(output: str, field: str) -> Optional[int]:
    """Scan backwards for the newest line carrying ``field``, ignoring noise.

    The transport hands back stdout AND stderr concatenated, and the launcher
    writes to stderr: ``uv`` emits a deprecation warning on every invocation, so
    the literal last line is that warning and not the probe's answer. Taking the
    last line blind made a healthy, empty dump root read as an unparsable disk
    and blocked the launch.
    """
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and field in doc:
            try:
                return int(doc[field])
            except (TypeError, ValueError):
                return None
    return None


def _split_endpoint(spec: str) -> tuple[str, int]:
    host, port = spec.rsplit(":", 1)
    return host, int(port)


def _next_version(current: str) -> str:
    from exp.rl_router.train_router import _next_version as bump

    return bump(current)


if __name__ == "__main__":
    main()
