"""Three-source batch assembly and the training-accepted manifest (X14 §3.6/§3.7).

An RL batch spans two machines. The server (wls) owns the feature shards and
their per-step sidecars; the conductor (t107) owns the journal slice and the
client per-step rows. Nothing may enter an Adam step until all three agree, on
every step, for a full batch of N episodes.

Join key
--------
``(task_uid, attempt, batch_id, weights_version, decision_idx)``.

``decision_idx`` is the server's per-episode verdict counter. The client's own
``step_idx`` is the physical env step and advances by ``replan_steps`` between
inference calls, so it cannot index verdicts; continuity ("0..K-1, no holes") is
only defined on ``decision_idx``.

Why a manifest and not the journal
----------------------------------
"Journal says terminal" does not imply "shard was finalized" — the episode-end
broadcast can fail after the journal record is written. The shard manifest is
therefore the authority on completeness, and the journal contributes the two
things only the scheduler knows: whether the result was *accepted* (not a stale
attempt from a superseded dispatch) and whether it carried an error.

Selection
---------
For each expected init slot exactly one attempt is marked ``training_selected``,
by a deterministic priority (the original uid first, then repair rounds in
ascending order). Everything else that was accepted is marked ``superseded``.
The trainer consumes ``training_selected`` and nothing else, so a repair round
can never double-count an init.

Transport
---------
The fleet does not share a mount: the conductor runs on t107, the shards live
on wls, and a package crosses by scp over the tether-exposed ssh port. The push
writes payload files first and ``COMPLETE.json`` **last**, so an interrupted
copy leaves no marker and is rejected rather than mistaken for a short batch;
re-pushing identical content is free, and a second batch claiming a delivered
id is refused outright. The three-source join then runs **on the server**,
because the shard manifest never leaves the machine that wrote it, and only the
``missing_slots`` list travels back to drive a repair round.

Usage::

    # on the conductor (t107), after a batch barrier
    uv run exp/rl_router/batch_package.py assemble --batch-id b3 --weights-version v3 \\
        --journal <journal.jsonl> --client-rows <rows.jsonl> \\
        --expected-slots <uids.txt> --out <dir>
    uv run exp/rl_router/batch_package.py push --package <dir> --remote <remote dir> \\
        --host <wls> --port 14024
    # on the server (wls)
    uv run exp/rl_router/batch_package.py verify --package <dir>
    uv run exp/rl_router/batch_package.py manifest --package <dir> --shards <dir> --out <path>
    # after the trainer checkpoint lands
    uv run exp/rl_router/batch_package.py reclaim --shards <dir> --checkpoint <ck.pt>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# The five identity fields every source must agree on, plus the step coordinate.
JOIN_KEYS = ("task_uid", "attempt", "batch_id", "weights_version", "decision_idx")

# Episode identity without the step coordinate.
EPISODE_KEYS = ("task_uid", "attempt", "batch_id", "weights_version")

# A repair round re-runs the same init under a new uid so the two attempts never
# contend for one shard path. ``<orig>#r<n>``.
_REPAIR_RE = re.compile(r"^(?P<orig>.+)#r(?P<round>\d+)$")

# Written by the conductor once every file of a package has landed; its absence
# is how a partially-copied package is detected.
COMPLETE_MARKER = "COMPLETE.json"

PACKAGE_FILES = (
    "journal_slice.jsonl",
    "accepted_manifest.json",
    "per_step_rows_batch.jsonl",
)


# ---------------------------------------------------------------------------
# Slot identity
# ---------------------------------------------------------------------------


def slot_of(task_uid: str) -> str:
    """Map a (possibly repaired) task uid back to the init slot it fills."""
    m = _REPAIR_RE.match(task_uid)
    return m.group("orig") if m else task_uid


def repair_round(task_uid: str) -> int:
    """Repair generation of a uid; 0 for an original dispatch.

    Doubles as the selection priority, which is why it is a total order rather
    than a boolean: with two repair rounds the earlier one wins, deterministically.
    """
    m = _REPAIR_RE.match(task_uid)
    return int(m.group("round")) if m else 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_jsonl(path: str | pathlib.Path) -> list[dict]:
    """Read a JSONL file, tolerating a torn final line from a hard crash."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_shard_manifest(shard_dir: str | pathlib.Path) -> list[dict]:
    """Load the server-side completion ledger for one ``<run_id>/<batch_id>``."""
    return read_jsonl(pathlib.Path(shard_dir) / "manifest.jsonl")


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    """One (uid, attempt) candidate, joined across all three sources."""

    task_uid: str
    attempt: int
    batch_id: str
    weights_version: str
    slot: str
    repair_round: int
    success: bool
    rows: int
    dim: int
    shard: Optional[str]
    sidecar: Optional[str]
    sha256: Optional[str]
    sidecar_sha256: Optional[str] = None

    @property
    def key(self) -> tuple:
        return tuple(getattr(self, k) for k in EPISODE_KEYS)


@dataclasses.dataclass
class BatchManifest:
    """The batch's admission decision, machine-checkable and reproducible."""

    batch_id: str
    weights_version: str
    expected_slots: list[str]
    selected: list[EpisodeRecord]
    superseded: list[dict]
    rejected: list[dict]
    missing_slots: list[str]

    @property
    def complete(self) -> bool:
        """A batch is trainable only at its full frozen size."""
        return not self.missing_slots

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "weights_version": self.weights_version,
            "expected_slots": list(self.expected_slots),
            "training_selected": [dataclasses.asdict(r) for r in self.selected],
            "superseded": list(self.superseded),
            "rejected": list(self.rejected),
            "missing_slots": list(self.missing_slots),
            "complete": self.complete,
        }


def build_batch_manifest(
    *,
    batch_id: str,
    weights_version: str,
    expected_slots: Iterable[str],
    journal: list[dict],
    client_rows: list[dict],
    shards: list[dict],
    quarantine: Optional[Iterable[tuple[str, int]]] = None,
    shard_dir: Optional[str | pathlib.Path] = None,
    require_sidecar: bool = False,
) -> BatchManifest:
    """Join the three sources and pick exactly one attempt per expected slot.

    An episode is admissible only when every one of these holds:

      - the scheduler accepted the result (``accepted is True``) — a stale
        attempt from a superseded dispatch is journaled identically otherwise;
      - it carried no error (a sidecar failure means the arm did not execute);
      - its shard is ``complete`` (``partial`` means the episode never reached
        its end) and the weights version matches the batch's;
      - the server and the client saw the same verdicts: the dumped rows and the
        client's ``decision_idx`` sequence agree and are dense ``0..K-1``;
      - it is not ``quarantine``d — the trainer's on-policy admission rejected
        that exact ``(task_uid, attempt)`` in an earlier round, and without this
        the deterministic priority would re-select the same bad attempt forever
        and the repair loop could never converge.

    Each rejection is recorded with its reason so a batch that fails to fill is
    diagnosable without re-running it.
    """
    expected = list(dict.fromkeys(expected_slots))
    if require_sidecar and shard_dir is None:
        raise ValueError(
            "production admission requires shard_dir: without it the sidecar — the "
            "behaviour authority the gradient is weighted by — is never inspected, "
            "and a malformed one reaches the trainer as an unrepairable crash"
        )
    banned = {(str(u), int(a)) for u, a in (quarantine or ())}
    # Indexed by (uid, attempt) only, with batch_id / weights_version checked
    # explicitly below: folding them into the key would report a shard written
    # under the wrong version as simply "missing", hiding a hot-swap race behind
    # the same reason as a lost episode.
    shards_by_dispatch: dict[tuple[str, int], list[dict]] = {}
    for s in shards:
        shards_by_dispatch.setdefault(
            (str(s.get("task_uid")), int(s.get("attempt", -1))), []
        ).append(s)
    client_by_key = _client_decisions(client_rows)

    rejected: list[dict] = []
    candidates: dict[str, list[EpisodeRecord]] = {slot: [] for slot in expected}

    for entry in journal:
        uid = str(entry.get("task_uid", ""))
        slot = slot_of(uid)
        if slot not in candidates:
            continue  # belongs to another batch / another stage
        attempt = int(entry.get("attempt", -1))

        def reject(reason: str) -> None:
            rejected.append({"task_uid": uid, "attempt": attempt, "reason": reason})

        if (uid, attempt) in banned:
            reject("quarantined_by_trainer_admission")
            continue
        if entry.get("accepted") is not True:
            reject("scheduler_rejected")           # stale / duplicate dispatch
            continue
        if entry.get("error") is not None:
            reject("episode_error")
            continue
        pool = shards_by_dispatch.get((uid, attempt), [])
        if not pool:
            reject("shard_missing")
            continue
        shard = next(
            (s for s in pool
             if str(s.get("weights_version")) == weights_version
             and str(s.get("batch_id")) == batch_id),
            None,
        )
        if shard is None:
            reject("weights_version_mismatch")
            continue
        if shard.get("status") != "complete":
            reject(f"shard_{shard.get('status')}")
            continue
        rows = int(shard.get("rows", 0))
        if rows <= 0:
            reject("no_steps")
            continue
        client_idx = client_by_key.get((uid, attempt, weights_version))
        if client_idx is None:
            reject("client_rows_missing")
            continue
        if client_idx != list(range(rows)):
            reject("decision_idx_discontinuous")
            continue
        # The sidecar is the third source and it lives here, next to the shard.
        # Validating it during admission is what makes a bad behaviour record a
        # REPAIRABLE missing slot; discovering it later, inside the trainer's
        # loader, turns it into a generic crash that the bounded repair loop
        # cannot act on.
        if shard_dir is not None:
            reason = sidecar_defect(pathlib.Path(shard_dir), shard, expected_rows=rows)
            if reason is not None:
                reject(reason)
                continue
        candidates[slot].append(EpisodeRecord(
            task_uid=uid, attempt=attempt, batch_id=batch_id,
            weights_version=weights_version, slot=slot,
            repair_round=repair_round(uid), success=bool(entry.get("success")),
            rows=rows, dim=int(shard.get("dim", 0)),
            shard=shard.get("shard"), sidecar=shard.get("sidecar"),
            sha256=shard.get("sha256"), sidecar_sha256=shard.get("sidecar_sha256"),
        ))

    selected: list[EpisodeRecord] = []
    superseded: list[dict] = []
    missing: list[str] = []
    for slot in expected:
        pool = candidates[slot]
        if not pool:
            missing.append(slot)
            continue
        # Deterministic priority: the original dispatch, then repair rounds in
        # order, then the highest attempt of that uid. Reproducible from the
        # inputs alone — no wall-clock, no arrival order.
        pool.sort(key=lambda r: (r.repair_round, -r.attempt))
        selected.append(pool[0])
        superseded.extend(
            {"task_uid": r.task_uid, "attempt": r.attempt, "slot": slot} for r in pool[1:]
        )

    return BatchManifest(
        batch_id=batch_id, weights_version=weights_version, expected_slots=expected,
        selected=selected, superseded=superseded, rejected=rejected, missing_slots=missing,
    )


def sidecar_defect(
    shard_dir: pathlib.Path, shard: dict, *, expected_rows: int
) -> Optional[str]:
    """Return a rejection reason for this episode's sidecar, or None.

    Same three gates the trainer's loader applies, run here where they can still
    be repaired: the file must be the one the manifest declared (digest), every
    row must belong to this exact ``(task_uid, attempt, batch_id,
    weights_version)``, and ``decision_idx`` must be dense ``0..K-1``. Duplicates
    and holes both yield a correctly-sized list, so a row count cannot see them,
    and a repeated index would double-weight one step's gradient.
    """
    name = shard.get("sidecar")
    if not name:
        return "sidecar_missing"
    path = shard_dir / str(name)
    if not path.exists():
        return "sidecar_missing"
    raw = path.read_bytes()
    declared = shard.get("sidecar_sha256")
    if declared is not None and hashlib.sha256(raw).hexdigest() != declared:
        return "sidecar_digest_mismatch"
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "sidecar_unreadable"
    # Every check below is defensive about SHAPE as well as value: a syntactically
    # valid JSON line can still be a list, or carry a string where a float
    # belongs. Anything that would raise later in the trainer's loader has to be
    # a named rejection here, or it escapes the bounded repair loop as a generic
    # crash.
    if not all(isinstance(row, dict) for row in rows):
        return "sidecar_row_not_an_object"
    identity = (str(shard.get("task_uid")), int(shard.get("attempt", -1)),
                str(shard.get("batch_id")), str(shard.get("weights_version")))
    for row in rows:
        for key in ("task_uid", "attempt", "batch_id", "weights_version",
                    "decision_idx", "arm_sampled", "arm_mapped", "logits",
                    "logprob_sampled"):
            if key not in row:
                return "sidecar_row_missing_field"
        try:
            actual = (str(row["task_uid"]), int(row["attempt"]),
                      str(row["batch_id"]), str(row["weights_version"]))
        except (TypeError, ValueError):
            return "sidecar_identity_malformed"
        if actual != identity:
            return "sidecar_identity_mismatch"
        logits = row["logits"]
        if not isinstance(logits, list) or not logits:
            return "sidecar_logits_malformed"
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in logits):
            return "sidecar_logits_malformed"
        if not isinstance(row["logprob_sampled"], (int, float)) or \
                isinstance(row["logprob_sampled"], bool):
            return "sidecar_logprob_malformed"
        if not isinstance(row["arm_sampled"], str) or not isinstance(row["arm_mapped"], str):
            return "sidecar_arm_malformed"
    widths = {len(row["logits"]) for row in rows}
    if len(widths) > 1:
        return "sidecar_logits_ragged"
    try:
        indices = sorted(int(r["decision_idx"]) for r in rows)
    except (TypeError, ValueError):
        return "sidecar_decision_idx_malformed"
    if indices != list(range(expected_rows)):
        return "sidecar_decision_idx_discontinuous"
    return None


def _client_decisions(client_rows: list[dict]) -> dict[tuple, list[int]]:
    """Per-episode sorted ``decision_idx`` list from the client per-step rows.

    Keyed on ``(task_uid, attempt, weights_version)`` rather than the full
    five-part identity: ``batch_id`` is deliberately absent from the
    ``router_outputs`` wire schema (it is not a per-verdict quantity), and a
    package is one batch by construction, so the batch is already pinned by the
    context these rows arrive in.
    """
    out: dict[tuple, list[int]] = {}
    for row in client_rows:
        if row.get("_kind") is not None:
            continue  # episode_summary / client_timing provenance rows
        ro = row.get("router_outputs")
        if not ro:
            continue
        key = (str(row.get("task_uid")), int(row.get("attempt", -1)),
               str(ro.get("weights_version")))
        out.setdefault(key, []).append(int(ro["decision_idx"]))
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Package protocol
# ---------------------------------------------------------------------------


def assemble_package(
    out_dir: str | pathlib.Path,
    *,
    batch_id: str,
    weights_version: str,
    journal_rows: list[dict],
    client_rows: list[dict],
    expected_slots: Iterable[str],
    quarantine: Optional[Iterable[tuple[str, int]]] = None,
) -> pathlib.Path:
    """Write an immutable batch package, completion marker last.

    The marker carries the per-file digests and is written only after every
    payload file is on disk, so a package interrupted mid-copy is detectable
    (and re-pushable idempotently) rather than silently short.

    ``quarantine`` travels with the package because the join runs remotely: the
    attempts the trainer's on-policy check already rejected have to be excluded
    there, or the deterministic priority would re-select them and the repair
    loop could never converge.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "journal_slice.jsonl", journal_rows)
    _write_jsonl(out / "per_step_rows_batch.jsonl", client_rows)
    (out / "accepted_manifest.json").write_text(
        json.dumps({
            "batch_id": batch_id,
            "weights_version": weights_version,
            "expected_slots": list(dict.fromkeys(expected_slots)),
            "quarantine": [[str(u), int(a)] for u, a in (quarantine or ())],
        }, indent=2),
        encoding="utf-8",
    )
    digests = {name: sha256_file(out / name) for name in PACKAGE_FILES}
    (out / COMPLETE_MARKER).write_text(
        json.dumps({
            "batch_id": batch_id,
            "weights_version": weights_version,
            "sha256": digests,
            "package_sha256": hashlib.sha256(
                json.dumps(digests, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }, indent=2),
        encoding="utf-8",
    )
    return out


def verify_package(package_dir: str | pathlib.Path) -> dict:
    """Validate a received package. Raises ``ValueError`` on any defect."""
    pkg = pathlib.Path(package_dir)
    marker = pkg / COMPLETE_MARKER
    if not marker.exists():
        raise ValueError(
            f"batch package {pkg} has no {COMPLETE_MARKER}: the copy never finished "
            "(re-push is idempotent)"
        )
    meta = json.loads(marker.read_text(encoding="utf-8"))
    for name in PACKAGE_FILES:
        path = pkg / name
        if not path.exists():
            raise ValueError(f"batch package {pkg} is missing {name}")
        actual = sha256_file(path)
        expected = meta.get("sha256", {}).get(name)
        if actual != expected:
            raise ValueError(
                f"batch package {pkg}: {name} sha256 {actual} != recorded {expected}"
            )
    return meta


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Cross-machine transport (t107 conductor -> wls server)
#
# The fleet is deliberately NOT sharing a mount: the conductor runs on t107 and
# the shards live on wls, so a package crosses by scp over the wls-ssh tunnel.
# Everything below is written against a Transport seam so the protocol can be
# exercised end to end without a second machine.
# ---------------------------------------------------------------------------


class TransportError(RuntimeError):
    """A push / remote command failed after its retry budget."""


class LocalTransport:
    """Same-filesystem transport — single-host runs and tests."""

    def __init__(self, workdir: str | pathlib.Path = ".") -> None:
        self.workdir = str(workdir)

    def push_dir(self, local: pathlib.Path, remote: str, *, names: list[str]) -> None:
        target = pathlib.Path(remote)
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(local / name, target / name)

    def push_file(self, local: pathlib.Path, remote: str) -> None:
        """Place ``local`` AT ``remote`` — destination basename honoured, atomically."""
        target = pathlib.Path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".uploading")
        shutil.copyfile(local, tmp)
        os.replace(tmp, target)

    def fetch_file(self, remote: str, local: pathlib.Path) -> bool:
        source = pathlib.Path(remote)
        if not source.exists():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)
        return True

    def run(self, command: str) -> tuple[int, str]:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


class SshTransport:
    """scp / ssh over the tether-exposed port (plan §4.3: ``wls-ssh :14024``)."""

    def __init__(self, host: str, *, port: int, user: str = "", workdir: str = ".") -> None:
        self.host = host
        self.port = int(port)
        self.user = user
        self.workdir = workdir

    @property
    def _target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def push_dir(self, local: pathlib.Path, remote: str, *, names: list[str]) -> None:
        self._check(f"ssh -p {self.port} {self._target} mkdir -p {shlex.quote(remote)}")
        for name in names:
            src = shlex.quote(str(local / name))
            dst = f"{self._target}:{shlex.quote(remote + '/' + name)}"
            self._check(f"scp -P {self.port} {src} {dst}")

    def push_file(self, local: pathlib.Path, remote: str) -> None:
        """Place ``local`` AT ``remote``, keeping the DESTINATION basename.

        ``scp`` to a ``.uploading`` sibling and then rename: the destination
        name is what the fleet reads (``weights/v3.pt``), and it is rarely the
        source's name (``warmstart_l10.pt``). Pushing by source basename put the
        file somewhere nothing looked for it, and the first batch then failed to
        find its policy. The rename also makes a half-copied upload invisible.
        """
        parent = remote.rsplit("/", 1)[0]
        self._check(f"ssh -p {self.port} {self._target} mkdir -p {shlex.quote(parent)}")
        staging = f"{remote}.uploading"
        self._check(
            f"scp -P {self.port} {shlex.quote(str(local))} "
            f"{self._target}:{shlex.quote(staging)}"
        )
        self._check(
            f"ssh -p {self.port} {self._target} mv {shlex.quote(staging)} {shlex.quote(remote)}"
        )

    def fetch_file(self, remote: str, local: pathlib.Path) -> bool:
        local.parent.mkdir(parents=True, exist_ok=True)
        code, _ = self.run(f"test -f {shlex.quote(remote)}")
        if code != 0:
            return False
        self._check(
            f"scp -P {self.port} {self._target}:{shlex.quote(remote)} {shlex.quote(str(local))}"
        )
        return True

    def run(self, command: str) -> tuple[int, str]:
        wrapped = f"ssh -p {self.port} {self._target} {shlex.quote(command)}"
        proc = subprocess.run(wrapped, shell=True, capture_output=True, text=True, check=False)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def _check(self, command: str) -> None:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise TransportError(f"{command} -> exit {proc.returncode}: {proc.stderr.strip()}")


def push_package(
    transport,
    local_dir: str | pathlib.Path,
    remote_dir: str,
    *,
    retries: int = 3,
) -> str:
    """Deliver a package, marker last. Returns ``"pushed"`` or ``"already_delivered"``.

    Ordering is the whole protocol: payload files first, ``COMPLETE.json`` last.
    A copy interrupted anywhere leaves the remote without a marker, which
    ``verify_package`` rejects, so a partial delivery can never be mistaken for
    a short batch. Re-pushing identical content is therefore free.

    A remote marker for the same ``batch_id`` with a *different* digest is
    refused outright: two different batches claiming one id would have the
    trainer's idempotence ledger silently skip the second.
    """
    local = pathlib.Path(local_dir)
    marker = json.loads((local / COMPLETE_MARKER).read_text(encoding="utf-8"))
    remote_marker = _read_remote_marker(transport, remote_dir)
    if remote_marker is not None:
        if remote_marker.get("package_sha256") == marker["package_sha256"]:
            return "already_delivered"
        raise TransportError(
            f"remote {remote_dir} already holds batch "
            f"{remote_marker.get('batch_id')!r} with a different digest "
            f"({remote_marker.get('package_sha256')} != {marker['package_sha256']}); "
            "refusing to overwrite a delivered batch"
        )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            transport.push_dir(local, remote_dir, names=list(PACKAGE_FILES))
            transport.push_dir(local, remote_dir, names=[COMPLETE_MARKER])
            return "pushed"
        except Exception as exc:  # noqa: BLE001 - retried, then escalated
            last_error = exc
            logger.warning("package push attempt %d/%d failed: %s", attempt, retries, exc)
    raise TransportError(
        f"ALERT: package push to {remote_dir} failed after {retries} attempts: {last_error}"
    )


def _read_remote_marker(transport, remote_dir: str) -> Optional[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        local = pathlib.Path(tmp) / COMPLETE_MARKER
        if not transport.fetch_file(f"{remote_dir}/{COMPLETE_MARKER}", local):
            return None
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


def remote_build_manifest(
    transport,
    *,
    remote_package: str,
    remote_shards: str,
    remote_manifest: str,
    python: str = "uv run",
    workdir: str = ".",
) -> tuple[bool, str]:
    """Run the three-source join **where the shards are** (the server).

    Returns ``(complete, output)``. The join has to happen remotely: the shard
    manifest is the completeness authority and it never leaves the machine that
    wrote it.
    """
    cmd = (
        f"cd {shlex.quote(workdir)} && {python} exp/rl_router/batch_package.py manifest "
        f"--package {shlex.quote(remote_package)} --shards {shlex.quote(remote_shards)} "
        f"--out {shlex.quote(remote_manifest)}"
    )
    code, output = transport.run(cmd)
    return code == 0, output


def fetch_missing_slots(
    transport, *, remote_manifest: str, local_path: str | pathlib.Path
) -> list[str]:
    """Pull the remote manifest back and return the slots still to repair."""
    local = pathlib.Path(local_path)
    if not transport.fetch_file(remote_manifest, local):
        raise TransportError(f"remote manifest {remote_manifest} was not produced")
    doc = json.loads(local.read_text(encoding="utf-8"))
    return list(doc.get("missing_slots", []))


# ---------------------------------------------------------------------------
# Post-checkpoint shard reclamation (§3.0 capacity budget)
# ---------------------------------------------------------------------------


def trainer_consumed(checkpoint_path: str | pathlib.Path) -> list[dict]:
    """The consumed-batch ledger recorded inside a trainer checkpoint."""
    import torch

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return list(blob.get("consumed_batches", []))


def reclaim_batch_shards(
    shard_dir: str | pathlib.Path,
    *,
    checkpoint_path: str | pathlib.Path,
    batch_id: str,
    package_sha256: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Delete THIS batch's ``.bin`` shards; keep every audit artifact.

    The gate is the checkpoint's **consumed ledger**, not the checkpoint's mere
    existence: a stale checkpoint from an earlier batch is a file on disk too,
    and letting it authorise a deletion would destroy rollouts whose update
    never happened. The ledger names the batch that was applied, and (when
    given) the package digest it was applied from, so the deletion can only
    follow the update it belongs to.

    Sidecars and the manifest stay forever — they are kilobytes and they are the
    audit trail — while the features are ~26 MB/episode and are what would
    otherwise grow to ~105 GB over a 4k-episode run instead of a ~5-6 GB
    steady state.
    """
    checkpoint = pathlib.Path(checkpoint_path)
    if not checkpoint.exists():
        raise ValueError(
            f"refusing to reclaim {shard_dir}: trainer checkpoint {checkpoint} does not "
            "exist, so the batch's update is not proven durable"
        )
    consumed = trainer_consumed(checkpoint)
    entry = next((c for c in consumed if c.get("batch_id") == batch_id), None)
    if entry is None:
        raise ValueError(
            f"refusing to reclaim {shard_dir}: checkpoint {checkpoint} has not consumed "
            f"batch {batch_id!r} (its ledger holds "
            f"{[c.get('batch_id') for c in consumed]}); deleting now would destroy "
            "rollouts whose update never landed"
        )
    if package_sha256 is not None and entry.get("package_sha256") != package_sha256:
        raise ValueError(
            f"refusing to reclaim {shard_dir}: batch {batch_id!r} was consumed from "
            f"package {entry.get('package_sha256')!r}, not {package_sha256!r}"
        )
    directory = pathlib.Path(shard_dir)
    freed, removed = 0, []
    for shard in sorted(directory.glob("*.bin")):
        freed += shard.stat().st_size
        removed.append(shard.name)
        if not dry_run:
            shard.unlink()
    # Torn writes from a crashed process are reclaimable too.
    for tmp in sorted(directory.glob("*.bin.tmp")):
        freed += tmp.stat().st_size
        removed.append(tmp.name)
        if not dry_run:
            tmp.unlink()
    return {
        "shard_dir": str(directory),
        "batch_id": batch_id,
        "removed": removed,
        "bytes_freed": freed,
        "kept": ["manifest.jsonl", "*.jsonl sidecars"],
        "dry_run": dry_run,
    }


def steady_state_bytes(dump_root: str | pathlib.Path) -> int:
    """Total live feature bytes under a dump root (the capacity gate's metric)."""
    return sum(p.stat().st_size for p in pathlib.Path(dump_root).rglob("*.bin"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_verify(args) -> None:
    meta = verify_package(args.package)
    print(json.dumps(meta, indent=2))


def _cmd_manifest(args) -> None:
    pkg = pathlib.Path(args.package)
    verify_package(pkg)
    accepted = json.loads((pkg / "accepted_manifest.json").read_text(encoding="utf-8"))
    manifest = build_batch_manifest(
        batch_id=accepted["batch_id"],
        weights_version=accepted["weights_version"],
        expected_slots=accepted["expected_slots"],
        journal=read_jsonl(pkg / "journal_slice.jsonl"),
        client_rows=read_jsonl(pkg / "per_step_rows_batch.jsonl"),
        shards=load_shard_manifest(args.shards),
        quarantine=[tuple(q) for q in accepted.get("quarantine", [])],
        shard_dir=args.shards,          # the join runs where the sidecars live
        require_sidecar=True,           # production path: never skip it
    )
    payload = json.dumps(manifest.to_dict(), indent=2)
    if args.out:
        pathlib.Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    if not manifest.complete:
        raise SystemExit(
            f"batch {manifest.batch_id} is short {len(manifest.missing_slots)} slot(s); "
            "dispatch a repair round before training"
        )


def _cmd_assemble(args) -> None:
    expected = [
        line.strip()
        for line in pathlib.Path(args.expected_slots).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = assemble_package(
        args.out, batch_id=args.batch_id, weights_version=args.weights_version,
        journal_rows=read_jsonl(args.journal), client_rows=read_jsonl(args.client_rows),
        expected_slots=expected,
    )
    print(json.dumps({"package": str(out), "expected_slots": len(expected)}, indent=2))


def _cmd_push(args) -> None:
    transport = (
        LocalTransport() if args.local
        else SshTransport(args.host, port=args.port, user=args.user)
    )
    status = push_package(transport, args.package, args.remote, retries=args.retries)
    print(json.dumps({"status": status, "remote": args.remote}, indent=2))


def _cmd_capacity(args) -> None:
    print(json.dumps({"root": args.root, "live_bytes": steady_state_bytes(args.root)}))


def _cmd_reclaim(args) -> None:
    report = reclaim_batch_shards(
        args.shards, checkpoint_path=args.checkpoint, batch_id=args.batch_id,
        package_sha256=args.package_sha256 or None, dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_assemble = sub.add_parser("assemble", help="build an immutable batch package")
    p_assemble.add_argument("--batch-id", required=True)
    p_assemble.add_argument("--weights-version", required=True)
    p_assemble.add_argument("--journal", required=True)
    p_assemble.add_argument("--client-rows", required=True)
    p_assemble.add_argument("--expected-slots", required=True, help="one task_uid per line")
    p_assemble.add_argument("--out", required=True)
    p_assemble.set_defaults(func=_cmd_assemble)

    p_push = sub.add_parser("push", help="deliver a package to the server, marker last")
    p_push.add_argument("--package", required=True)
    p_push.add_argument("--remote", required=True)
    p_push.add_argument("--host", default="")
    p_push.add_argument("--port", type=int, default=22)
    p_push.add_argument("--user", default="")
    p_push.add_argument("--local", action="store_true", help="same-host copy")
    p_push.add_argument("--retries", type=int, default=3)
    p_push.set_defaults(func=_cmd_push)

    p_reclaim = sub.add_parser("reclaim", help="delete a consumed batch's feature shards")
    p_reclaim.add_argument("--shards", required=True)
    p_reclaim.add_argument("--checkpoint", required=True)
    p_reclaim.add_argument("--batch-id", required=True,
                           help="the batch the checkpoint's ledger must name")
    p_reclaim.add_argument("--package-sha256", default="")
    p_reclaim.add_argument("--dry-run", action="store_true")
    p_reclaim.set_defaults(func=_cmd_reclaim)

    p_cap = sub.add_parser("capacity", help="live feature bytes under a dump root")
    p_cap.add_argument("--root", required=True)
    p_cap.set_defaults(func=_cmd_capacity)

    p_verify = sub.add_parser("verify", help="check a received package's marker + digests")
    p_verify.add_argument("--package", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_manifest = sub.add_parser("manifest", help="join three sources into a training manifest")
    p_manifest.add_argument("--package", required=True)
    p_manifest.add_argument("--shards", required=True, help="<dump_dir>/<run_id>/<batch_id>")
    p_manifest.add_argument("--out", default="")
    p_manifest.set_defaults(func=_cmd_manifest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
