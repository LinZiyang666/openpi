"""Shared manifests, byte identities and production retrieval for cache pruning.

Public helpers keep experiment artifacts immutable and fingerprinted. Retrieval
delegates to the existing CP1 strategy and backend; no inference code is changed.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Iterator

import numpy as np
import torch
import yaml

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    SearchContext,
    WeightedScoreSumKnnStrategy,
)
from openpi.cache.types import CheckpointID

VERSION = "cache_prune_v1"
SUITES = ("libero_spatial", "libero_10")
REGIMES = ("rit50", "cs500_success")
FIELDS = ("vision_0", "vision_1", "robot_state")
SEED = 20260911
ROOT = Path(__file__).resolve().parent
CENSUS = {
    ("libero_spatial", "rit50"): (49, 1018),
    ("libero_10", "rit50"): (50, 2640),
    ("libero_spatial", "cs500_success"): (439, 9329),
    ("libero_10", "cs500_success"): (392, 20461),
}


def task_prompt(task_name: str) -> str:
    """Remove LIBERO-10 scene identifiers from file names to recover task language."""
    return re.sub(r"^(?:LIVING_ROOM|KITCHEN|STUDY)_SCENE\d+_", "", task_name).replace(
        "_", " "
    )


@dataclass(frozen=True)
class SourceRow:
    """Original entry identity and progress, unaffected by subsequent pruning."""

    id: str
    source_ordinal: int
    trajectory_id: str
    task_key: str
    step_idx: int
    length: int
    remaining: int
    prev_ids: list[str]
    next_ids: list[str]


@dataclass(frozen=True)
class PruneSelection:
    """A source-ordered retained set and one direct witness per deletion."""

    threshold: float | None
    source_rows_digest: str
    kept_ids: list[str]
    witnesses: list[dict]

    def to_dict(self) -> dict:
        """Return a JSON-compatible selection with its retained-set identity."""
        value = asdict(self)
        value["keep_digest"] = digest(self.kept_ids)
        return value


def require(condition: bool, message: str) -> None:
    """Reject a violated experiment contract, including under Python -O."""
    if not condition:
        raise ValueError(message)


def digest(value) -> str:
    """Hash canonical JSON without platform-dependent float or NaN extensions."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256(path: str | Path) -> str:
    """Stream a file hash without materializing multi-gigabyte artifacts."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_identity(path: str | Path) -> dict:
    """Bind a resolved path to current byte size and SHA256."""
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_identity(identity: dict) -> None:
    """Re-read a file and reject changed bytes or path identity."""
    require(
        file_identity(identity["path"]) == identity,
        f"file identity changed: {identity['path']}",
    )


def seal(value: dict) -> dict:
    """Return a manifest with a self-consistent content digest."""
    body = {k: v for k, v in value.items() if k != "digest"}
    return {**body, "digest": digest(body)}


def check_seal(value: dict) -> None:
    """Reject truncated, edited or unknown-version experiment manifests."""
    require(value.get("version") == VERSION, "manifest version mismatch")
    require(value.get("digest") == seal(value)["digest"], "manifest digest mismatch")


def read_json(path: str | Path) -> dict:
    """Read a JSON document; callers validate its schema and seal."""
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: dict) -> None:
    """Create a JSON file exclusively, refusing any overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


@contextmanager
def new_directory(destination: str | Path) -> Iterator[Path]:
    """Publish an entire new artifact directory only after successful completion."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A sibling reservation also serializes cooperating producers of this path.
    reservation = destination.with_name(destination.name + ".lock")
    fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        require(not destination.exists(), f"output already exists: {destination}")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=destination.name + ".partial-", dir=destination.parent
            )
        )
        yield temporary
        require(
            not destination.exists(), f"output appeared during build: {destination}"
        )
        temporary.rename(destination)
    finally:
        os.close(fd)
        reservation.unlink()
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def environment() -> dict:
    """Record the numeric environment responsible for frozen score matrices."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "node": platform.node(),
        "machine": platform.machine(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
    }


def implementation_identity() -> list[dict]:
    """Fingerprint this experiment and the shared code actually used by its runner."""
    repository = ROOT.parents[2]
    files = set(ROOT.glob("*.py")) | set((ROOT / "analysis").glob("*.py"))
    for directory in (
        "src/openpi/cache",
        "src/openpi/conductor",
        "src/openpi/models_pytorch",
        "src/openpi/policies",
        "src/openpi/serving",
        "examples/libero",
    ):
        files.update((repository / directory).rglob("*.py"))
    files.update(
        repository / path
        for path in (
            "scripts/serve_policy.py",
            "src/openpi/training/config.py",
            "exp/ablation_study/cache_size/run_size_eval.py",
            "exp/common/conductor_journal.py",
            "uv.lock",
        )
    )
    return [file_identity(path) for path in sorted(files)]


def template(suite: str) -> dict:
    """Load the checked-in, fixed retrieval and pure-cache template."""
    require(suite in SUITES, f"unsupported suite {suite}")
    return yaml.safe_load((ROOT / "config" / f"search_{suite}.yaml").read_text())


def retrieval_digest(config: dict) -> str:
    """Fingerprint every cache setting except the one allowed preload path."""
    canonical = copy.deepcopy(config)
    canonical["backend"]["in_memory"]["preload_path"] = "__SOURCE__"
    return digest(canonical)


def validate_config(config: dict, suite: str) -> None:
    """Reject every change except preload_path relative to the fixed template."""
    require(
        retrieval_digest(config) == retrieval_digest(template(suite)),
        "cache config differs from frozen RIT retrieval / pure-cache contract",
    )


def runtime_entries(entries: list) -> list:
    """Make backend-safe shallow object copies without changing export objects."""
    result = []
    for original in entries:
        entry = copy.copy(original)
        entry.payload = copy.copy(original.payload)
        entry.query_keys = {
            k: torch.as_tensor(v).detach().cpu().float().contiguous()
            for k, v in original.query_keys.items()
        }
        entry.payload.action_chunk = (
            torch.as_tensor(original.payload.action_chunk)
            .detach()
            .cpu()
            .float()
            .contiguous()
        )
        result.append(entry)
    return result


def make_strategy(entries: list, config: dict, top_k: int = 1):
    """Build the real immutable backend and weighted CP1 search strategy."""
    backend = InMemoryBackend(config["backend"]["vector_dims"])
    storage = CacheStorage(backend)
    storage.batch_insert(runtime_entries(entries))
    backend.freeze()
    return strategy_for_storage(storage, config, top_k)


def strategy_for_storage(storage: CacheStorage, config: dict, top_k: int = 1):
    """Use the same production strategy with a loaded or newly built backend."""
    spec = config["checkpoints"]["cp1"]["search_strategy"]
    return WeightedScoreSumKnnStrategy(
        storage,
        top_k=top_k,
        step_filter=spec["step_filter"],
        trajectory_depth=1,
        task_scoped=True,
        fusion_weights={k: config["keys"][k]["weight"] for k in FIELDS},
        field_similarity=spec["field_similarity"],
        score_normalization=spec["score_normalization"],
    )


def context(entry) -> SearchContext:
    """Emit only production-enabled query fields, never disabled stored keys."""
    return SearchContext(
        {
            k: torch.as_tensor(entry.query_keys[k]).cpu().float().contiguous()
            for k in FIELDS
        },
        CheckpointID.CP1,
        entry.step_idx,
        entry.payload.task_key,
    )
