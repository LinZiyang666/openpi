"""Shared helpers for opus round-2 tools.

RULE 1 (owner, non-negotiable): nothing with init index 30-49 is ever kept.  Every reader in this
package drops records whose subset init is >= 30 *at parse time*, before any field other than the
identity is looked at, and raises if an identity cannot be parsed.  Fitting uses inits 0-19 only
and evaluation uses inits 20-29 only (``FIT_INITS`` / ``EVAL_INITS``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/weiland/projects/openpi")
RUNS = Path("/home/weiland/trace_runs/os_closed_loop")
STORE = Path("/home/weiland/trace_runs/offline_search_store")
DERIVED = STORE / "derived" / "r09_opus"
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out"

DISCOVERY_MAX = 30          # inits 0..29 only
FIT_INITS = tuple(range(0, 20))
EVAL_INITS = tuple(range(20, 30))
# Run roots that the owner forbids opening at all (any root whose name starts with one of these prefixes).
FORBIDDEN_ROOTS = ("r09_astra_holdout", "r09_astra_holdout2")
FORBIDDEN_PREFIXES = ("r09_astra_holdout", "r09_holdout")


def forbidden(name: str) -> bool:
    return any(str(name).startswith(p) for p in FORBIDDEN_PREFIXES)

PRICE = {"pi05": (0.152, 0.848), "groot": (0.148, 0.852)}
WRIST_LOOK = 0.0646


def parse_uid(uid: str):
    """'<arm>:eval:<task>:<init>' -> (task, init). Raises on malformed identity."""
    parts = str(uid).rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"unparseable task_uid {uid!r}")
    return int(parts[2]), int(parts[3])


def allowed(init: int) -> bool:
    return 0 <= int(init) < DISCOVERY_MAX


def check_root(root) -> Path:
    root = Path(root)
    if any(forbidden(p) for p in root.parts):
        raise PermissionError(f"forbidden run root {root}")
    return root


def iter_jsonl_discovery(path, uid_key="task_uid"):
    """Yield parsed JSONL records whose identity is a discovery init (0-29) only.

    The identity is extracted with a cheap string search first so that a forbidden record is never
    JSON-decoded as a whole.  Records without a task_uid are yielded only if ``uid_key`` is None.
    """
    path = Path(path)
    check_root(path)
    needle = f'"{uid_key}": "' if uid_key else None
    with path.open("rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                continue  # unterminated tail
            line = raw.decode("utf-8", "replace")
            if needle is None:
                yield json.loads(line)
                continue
            i = line.find(needle)
            if i < 0:
                needle2 = f'"{uid_key}":"'
                i = line.find(needle2)
                if i < 0:
                    continue
                j = i + len(needle2)
            else:
                j = i + len(needle)
            k = line.find('"', j)
            uid = line[j:k]
            try:
                _, init = parse_uid(uid)
            except ValueError:
                continue
            if not allowed(init):
                continue          # RULE 1: dropped before decoding
            yield json.loads(line)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def owner_ir(model, looks, calls, decisions, wrist_looks=0):
    v, m = PRICE[model]
    return (v * (looks - wrist_looks) + WRIST_LOOK * wrist_looks + m * calls) / max(decisions, 1)
