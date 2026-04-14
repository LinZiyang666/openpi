from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so openpi and exp packages are importable
# when running as `python -m exp.qdrant_step_knn.qdrant_step_knn_experiment` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import pickle
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import dataclasses

import h5py
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models

from openpi.cache.types import VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE
from openpi.cache.storage_types import QueryFilter, QuerySpec
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.backends.qdrant_backend import QdrantVectorStore, QdrantBackendConfig

from exp.qdrant_step_knn.qdrant_openpi_common import build_named_vectors
from exp.qdrant_step_knn.qdrant_openpi_common import load_step_record
from exp.qdrant_step_knn.qdrant_openpi_common import make_point_id
from exp.qdrant_step_knn.qdrant_openpi_common import named_vector_chunks_map
from exp.qdrant_step_knn.qdrant_openpi_common import resolve_h5_paths
from exp.qdrant_step_knn.qdrant_openpi_common import scan_dataset
from exp.qdrant_step_knn.qdrant_openpi_common import StepLocator
from exp.qdrant_step_knn.qdrant_openpi_common import StepRecord
from exp.qdrant_step_knn.qdrant_openpi_common import VISION_FIELDS

ALL_LOGICAL_KEYS = (
    "vision_0",
    "vision_1",
    "vision_2",
    "prompt_emb",
    "robot_state",
    "noise_action_1",
    "noise_action_2",
    "noise_action_3",
    "noise_action_4",
    "noise_action_5",
    "noise_action_6",
    "noise_action_7",
    "noise_action_8",
    "noise_action_9",
    "clean_action",
)


@dataclass(frozen=True)
class DBPointMeta:
    point_id: int
    source_file: str
    step_name: str
    step_idx: int
    task: str
    episode_id: int
    clean_action_flat: np.ndarray
    clean_action_norm: float


@dataclass(frozen=True)
class RankedResult:
    point_id: int
    weighted_score: float
    logical_scores: dict[str, float]
    clean_action_l2: float


@dataclass(frozen=True)
class CollectionRun:
    collection_name: str
    atomic_request_count: int
    candidate_count: int
    total_time_sec: float
    top_results: list[RankedResult]


@dataclass(frozen=True)
class DBCacheState:
    db_stats: Any
    db_index: dict[int, DBPointMeta]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-level Qdrant KNN experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config JSON file.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")
    return config


def get_config_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config field '{key}' must be a non-empty string path")
    return Path(value)


def make_client(config: dict[str, Any]) -> QdrantClient:
    qdrant_cfg = config.get("qdrant", {})
    if not isinstance(qdrant_cfg, dict):
        raise ValueError("Config field 'qdrant' must be an object")
    return QdrantClient(
        url=str(qdrant_cfg.get("url", "http://localhost:6333")),
        grpc_port=int(qdrant_cfg.get("grpc_port", 6334)),
        prefer_grpc=not bool(qdrant_cfg.get("prefer_http", False)),
        timeout=int(qdrant_cfg.get("request_timeout", 300)),
    )


def make_named_storage(
    config: dict[str, Any],
    db_stats,
    selected_keys: list[str],
    weights: dict[str, float],
) -> CacheStorage:
    """Build a CacheStorage backed by QdrantVectorStore for named-vector mode.

    vector_dims are derived from db_stats.  Multi-chunk fields (raw embeddings
    with dim > 65535) are not supported by the storage interface and are skipped
    with a warning; only single-vector fields are included.
    """
    from exp.qdrant_step_knn.qdrant_openpi_common import named_vector_chunks_map

    qdrant_cfg = config.get("qdrant", {})
    experiment_cfg = config.get("experiment", {})

    chunk_map = named_vector_chunks_map(db_stats)
    vector_dims: dict[str, int] = {
        field_name: sum(c.end - c.start for c in chunks)
        for field_name in selected_keys
        if (chunks := chunk_map.get(field_name))
    }

    if not vector_dims:
        raise RuntimeError(
            "No recognised fields found for storage interface. "
            "Check that selected_keys contains valid CACHE_QUERY_FIELDS entries."
        )

    top_k = int(experiment_cfg.get("top_k", 10))
    candidate_limit = int(experiment_cfg.get("candidate_limit", 50))

    backend_cfg = QdrantBackendConfig(
        url=str(qdrant_cfg.get("url", "http://localhost:6333")),
        collection_name=str(qdrant_cfg.get("named_collection", "openpi_steps_named")),
        vector_dims=vector_dims,
        prefer_grpc=not bool(qdrant_cfg.get("prefer_http", False)),
        grpc_port=int(qdrant_cfg.get("grpc_port", 6334)),
        request_timeout=int(qdrant_cfg.get("request_timeout", 300)),
        rrf_k=int(experiment_cfg.get("rrf_k", 60)),
        candidate_multiplier=max(1, candidate_limit // max(1, top_k)),
        fusion_weights=weights if len(weights) > 1 else None,
    )
    return CacheStorage(QdrantVectorStore(backend_cfg))


def ensure_rrf_support() -> None:
    if hasattr(models, "RrfQuery") and hasattr(models, "Rrf"):
        return
    try:
        client_version = importlib.metadata.version("qdrant-client")
    except importlib.metadata.PackageNotFoundError:
        client_version = "unknown"
    raise RuntimeError(
        "This script requires qdrant-client with RrfQuery/Rrf support for single-request "
        f"server-side fusion. Current qdrant-client={client_version}. "
        "Use the base environment or upgrade qdrant-client."
    )


def parse_selected_keys(key_config: dict[str, Any]) -> list[str]:
    unknown = [key for key in key_config if key not in ALL_LOGICAL_KEYS]
    if unknown:
        raise ValueError(f"Unknown keys: {unknown}")
    selected_keys = []
    for key in ALL_LOGICAL_KEYS:
        value = key_config.get(key, False)
        if isinstance(value, bool):
            enabled = value
        elif isinstance(value, dict):
            enabled = bool(value.get("enabled", False))
        else:
            raise ValueError(f"Key config for '{key}' must be bool or object")
        if enabled:
            selected_keys.append(key)
    if not selected_keys:
        raise ValueError("At least one key must be enabled in config['keys']")
    return selected_keys


def parse_weights(selected_keys: list[str], key_config: dict[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key in selected_keys:
        value = key_config[key]
        if isinstance(value, bool):
            weights[key] = 1.0
        elif isinstance(value, dict):
            weights[key] = float(value.get("weight", 1.0))
        else:
            raise ValueError(f"Key config for '{key}' must be bool or object")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive")
    return {key: value / total for key, value in weights.items()}


def build_filter(step_filter: str, query_step_idx: int, step_window: int) -> models.Filter | None:
    if step_filter == "all":
        return None
    if step_filter == "exact":
        return models.Filter(
            must=[models.FieldCondition(key="step_idx", match=models.MatchValue(value=query_step_idx))]
        )
    if step_filter == "window":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="step_idx",
                    range=models.Range(
                        gte=query_step_idx - step_window,
                        lte=query_step_idx + step_window,
                    ),
                )
            ]
        )
    raise ValueError(f"Unsupported step filter: {step_filter}")


def build_local_step_predicate(step_filter: str, query_step_idx: int, step_window: int):
    if step_filter == "all":
        return lambda _: True
    if step_filter == "exact":
        return lambda step_idx: step_idx == query_step_idx
    if step_filter == "window":
        return lambda step_idx: abs(step_idx - query_step_idx) <= step_window
    raise ValueError(f"Unsupported step filter: {step_filter}")


def build_query_filter(step_filter: str, query_step_idx: int, step_window: int) -> QueryFilter | None:
    """Storage-compatible equivalent of build_filter() for CacheStorage.search()."""
    if step_filter == "all":
        return None
    if step_filter == "exact":
        return QueryFilter(step_range=(query_step_idx, query_step_idx))
    if step_filter == "window":
        return QueryFilter(step_range=(query_step_idx - step_window, query_step_idx + step_window))
    raise ValueError(f"Unsupported step filter: {step_filter}")


def build_db_index(stats) -> dict[int, DBPointMeta]:
    index: dict[int, DBPointMeta] = {}
    from exp.qdrant_step_knn.qdrant_openpi_common import iter_step_records

    log(f"Building local DB index for {stats.total_steps} steps...")
    for count, record in enumerate(iter_step_records(stats), start=1):
        point_id = make_point_id(record.source_file_display, record.step_name)
        clean_action_flat = np.asarray(record.clean_action, dtype=np.float32).reshape(-1)
        clean_action_norm = float(np.linalg.norm(clean_action_flat))
        index[point_id] = DBPointMeta(
            point_id=point_id,
            source_file=record.source_file_display,
            step_name=record.step_name,
            step_idx=record.step_idx,
            task=record.task,
            episode_id=record.episode_id,
            clean_action_flat=clean_action_flat,
            clean_action_norm=clean_action_norm,
        )
        if count % 100 == 0 or count == stats.total_steps:
            log(f"  Indexed {count}/{stats.total_steps} DB steps")
    log("Finished building local DB index")
    return index


def dataset_fingerprint(data_dir: Path) -> tuple[str, int]:
    resolved = data_dir.expanduser().resolve()
    h5_paths = resolve_h5_paths(resolved)
    digest = hashlib.sha256()
    digest.update(str(resolved).encode("utf-8"))
    for path in h5_paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest(), len(h5_paths)


def cache_settings(config: dict[str, Any]) -> tuple[bool, Path, bool]:
    cache_cfg = config.get("cache", {})
    if not isinstance(cache_cfg, dict):
        raise ValueError("Config field 'cache' must be an object")
    enabled = bool(cache_cfg.get("enabled", True))
    cache_dir = Path(str(cache_cfg.get("dir", ".cache/qdrant_step_knn")))
    refresh = bool(cache_cfg.get("refresh", False))
    return enabled, cache_dir, refresh


def cache_file_path(cache_dir: Path, data_dir: Path) -> Path:
    key = hashlib.sha1(str(data_dir.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir.expanduser().resolve() / f"db_state_{key}.pkl"


def load_or_build_db_state(config: dict[str, Any], db_data_dir: Path) -> DBCacheState:
    cache_version = 2
    enabled, cache_dir, refresh = cache_settings(config)
    if not enabled:
        log("Cache disabled: scanning DB data directory")
        db_stats = scan_dataset(db_data_dir)
        return DBCacheState(db_stats=db_stats, db_index=build_db_index(db_stats))

    fingerprint, h5_count = dataset_fingerprint(db_data_dir)
    cache_path = cache_file_path(cache_dir, db_data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not refresh and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if (
                isinstance(payload, dict)
                and payload.get("version") == cache_version
                and payload.get("fingerprint") == fingerprint
            ):
                log(f"Cache hit: loading DB state from {cache_path}")
                return DBCacheState(
                    db_stats=payload["db_stats"],
                    db_index=payload["db_index"],
                )
            log("Cache miss: fingerprint or cache version changed")
        except Exception as exc:
            log(f"Cache read failed, rebuilding: {exc}")

    log(
        f"Cache rebuild: scanning {h5_count} HDF5 files under {db_data_dir.expanduser().resolve()}"
    )
    db_stats = scan_dataset(db_data_dir)
    db_index = build_db_index(db_stats)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "version": cache_version,
                "fingerprint": fingerprint,
                "db_stats": db_stats,
                "db_index": db_index,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    log(f"Cache saved: {cache_path}")
    return DBCacheState(db_stats=db_stats, db_index=db_index)


def iter_query_records(query_file: Path, db_stats, requested_step_idxs: set[int], max_query_steps: int | None):
    query_path = query_file.expanduser().resolve()
    with h5py.File(query_path, "r") as h5_file:
        step_names = sorted(key for key in h5_file.keys() if key.startswith("step_"))
    yielded = 0
    for step_name in step_names:
        step_idx = int(step_name.split("_", maxsplit=1)[1])
        if requested_step_idxs and step_idx not in requested_step_idxs:
            continue
        yield load_step_record(StepLocator(source_file=query_path, step_name=step_name), db_stats)
        yielded += 1
        if max_query_steps is not None and yielded >= max_query_steps:
            break


def compute_background_clean_action_l2_mean(
    query_clean_action: np.ndarray,
    db_index: dict[int, DBPointMeta],
    predicate,
) -> float:
    total = 0.0
    count = 0
    for meta in db_index.values():
        if not predicate(meta.step_idx):
            continue
        total += float(np.linalg.norm(query_clean_action - meta.clean_action_flat))
        count += 1
    return total / count if count else 0.0


def compare_runs(named_run: CollectionRun | None, multivector_run: CollectionRun | None) -> dict[str, Any]:
    if named_run is None or multivector_run is None:
        return {
            "topk_same": None,
            "topk_overlap": None,
            "topk_jaccard": None,
        }
    named_ids = [result.point_id for result in named_run.top_results]
    multi_ids = [result.point_id for result in multivector_run.top_results]
    named_set = set(named_ids)
    multi_set = set(multi_ids)
    overlap = len(named_set & multi_set)
    union = len(named_set | multi_set)
    return {
        "topk_same": named_ids == multi_ids,
        "topk_overlap": overlap,
        "topk_jaccard": (overlap / union) if union else 1.0,
    }


def build_prefetches_for_collection(
    *,
    selected_keys: list[str],
    weights: dict[str, float],
    query_record: StepRecord,
    step_filter: models.Filter | None,
    candidate_limit: int,
    mode: str,
    db_stats,
) -> tuple[list[models.Prefetch], list[float], int]:
    prefetches: list[models.Prefetch] = []
    fusion_weights: list[float] = []
    atomic_request_count = 0

    if mode == "named":
        named_query_vectors = build_named_vectors(query_record, db_stats, fields=selected_keys)
        chunk_map = named_vector_chunks_map(db_stats)
        for key in selected_keys:
            chunk_names = [chunk.vector_name for chunk in chunk_map[key]]
            per_chunk_weight = weights[key] / len(chunk_names)
            for chunk_name in chunk_names:
                prefetches.append(
                    models.Prefetch(
                        query=named_query_vectors[chunk_name],
                        using=chunk_name,
                        filter=step_filter,
                        limit=candidate_limit,
                    )
                )
                fusion_weights.append(per_chunk_weight)
                atomic_request_count += 1
    else:
        # multivector mode removed
        raise ValueError(f"Unsupported mode {mode!r}. Only 'named' mode is supported.")
    return prefetches, fusion_weights, atomic_request_count


def run_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    query_record: StepRecord,
    db_index: dict[int, DBPointMeta],
    selected_keys: list[str],
    weights: dict[str, float],
    step_filter: models.Filter | None,
    top_k: int,
    candidate_limit: int,
    mode: str,
    db_stats,
    rrf_k: int,
) -> CollectionRun:
    log(f"  [{mode}] start server-side fused retrieval for {query_record.step_name} on {collection_name}")
    start = perf_counter()
    prefetches, fusion_weights, atomic_request_count = build_prefetches_for_collection(
        selected_keys=selected_keys,
        weights=weights,
        query_record=query_record,
        step_filter=step_filter,
        candidate_limit=candidate_limit,
        mode=mode,
        db_stats=db_stats,
    )
    log(
        f"  [{mode}] {query_record.step_name}: single query with {atomic_request_count} prefetch branches, "
        f"candidate_limit={candidate_limit}"
    )
    clean_action_query = np.asarray(query_record.clean_action, dtype=np.float32).reshape(-1)
    response = client.query_points(
        collection_name=collection_name,
        prefetch=prefetches,
        query=models.RrfQuery(rrf=models.Rrf(k=rrf_k, weights=fusion_weights)),
        limit=top_k,
        with_payload=False,
        with_vectors=False,
    )
    ranked_results: list[RankedResult] = []
    for point in response.points:
        point_id = int(point.id)
        meta = db_index[point_id]
        clean_action_l2 = float(np.linalg.norm(clean_action_query - meta.clean_action_flat))
        ranked_results.append(
            RankedResult(
                point_id=point_id,
                weighted_score=float(point.score),
                logical_scores={},
                clean_action_l2=clean_action_l2,
            )
        )

    total_time_sec = perf_counter() - start
    log(
        f"  [{mode}] done for {query_record.step_name}: "
        f"api_requests=1, prefetches={atomic_request_count}, topk_returned={len(response.points)}, "
        f"time={total_time_sec:.3f}s"
    )
    return CollectionRun(
        collection_name=collection_name,
        atomic_request_count=atomic_request_count,
        candidate_count=len(response.points),
        total_time_sec=total_time_sec,
        top_results=ranked_results,
    )


def run_named_collection(
    *,
    storage: CacheStorage,
    query_record: StepRecord,
    db_index: dict[int, DBPointMeta],
    selected_keys: list[str],
    top_k: int,
    query_filter: QueryFilter | None,
    step_filter_mode: str,
) -> CollectionRun:
    """Named-vector collection run using CacheStorage.search().

    The backend handles prefetch building, chunking, and RRF fusion internally.
    Fields not in CACHE_QUERY_FIELDS (noise_action_*, clean_action) are skipped.
    """
    collection_name = storage._backend._config.collection_name
    log(f"  [named] start storage search for {query_record.step_name} on {collection_name}")
    start = perf_counter()

    _field_inputs: dict[str, np.ndarray | None] = {
        VISION_0: query_record.vision.get("vision_0"),
        VISION_1: query_record.vision.get("vision_1"),
        VISION_2: query_record.vision.get("vision_2"),
        PROMPT_EMB: query_record.prompt_emb,
        ROBOT_STATE: query_record.robot_state,
    }
    supported = storage._backend.vector_dims
    query_keys = {
        field: np.asarray(arr, dtype=np.float32).reshape(-1)
        for field, arr in _field_inputs.items()
        if field in selected_keys and field in supported and arr is not None
    }

    spec = QuerySpec(query_keys=query_keys, top_k=top_k, filters=query_filter)
    results = storage.search(spec)

    if not results and step_filter_mode != "all":
        log(f"  [named] {query_record.step_name}: no results with filter, retrying without")
        results = storage.search(dataclasses.replace(spec, filters=None))

    atomic_request_count = len(query_keys)
    log(
        f"  [named] {query_record.step_name}: storage_fields={atomic_request_count}, "
        f"topk_returned={len(results)}"
    )

    clean_action_query = np.asarray(query_record.clean_action, dtype=np.float32).reshape(-1)
    ranked_results: list[RankedResult] = []
    for result in results:
        point_id = int(result.id)
        meta = db_index[point_id]
        clean_action_l2 = float(np.linalg.norm(clean_action_query - meta.clean_action_flat))
        ranked_results.append(RankedResult(
            point_id=point_id,
            weighted_score=result.score,
            logical_scores={},
            clean_action_l2=clean_action_l2,
        ))

    total_time_sec = perf_counter() - start
    log(
        f"  [named] done for {query_record.step_name}: "
        f"api_requests=1, prefetches={atomic_request_count}, topk_returned={len(results)}, "
        f"time={total_time_sec:.3f}s"
    )
    return CollectionRun(
        collection_name=collection_name,
        atomic_request_count=atomic_request_count,
        candidate_count=len(results),
        total_time_sec=total_time_sec,
        top_results=ranked_results,
    )


def compute_topk_clean_action_l2_mean(results: list[RankedResult]) -> float:
    if not results:
        return 0.0
    return float(sum(result.clean_action_l2 for result in results) / len(results))


def ensure_output_dir(path: Path | None) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("exp_outputs") / f"qdrant_step_knn_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown_report(
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> Path:
    report_path = output_dir / "report.md"
    lines: list[str] = []
    lines.append("# 实验报告")
    lines.append("")
    lines.append("## 配置")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config_payload, indent=2, ensure_ascii=False, sort_keys=True))
    lines.append("```")
    lines.append("")

    if summary_rows:
        lines.append("## 每个 Step 结果")
        lines.append("")
        lines.append("| query_step | named_topk_clean_action_l2_mean | multivector_topk_clean_action_l2_mean |")
        lines.append("| --- | ---: | ---: |")
        for row in summary_rows:
            lines.append(
                f"| {row['query_step_name']} | "
                f"{_fmt(row.get('named_topk_clean_action_l2_mean'))} | "
                f"{_fmt(row.get('multivector_topk_clean_action_l2_mean'))} |"
            )
        lines.append("")

        named_l2_mean = _mean([row.get("named_topk_clean_action_l2_mean") for row in summary_rows])
        multivector_l2_mean = _mean([row.get("multivector_topk_clean_action_l2_mean") for row in summary_rows])

        lines.append("## 平均结果")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | ---: |")
        lines.append(f"| named 平均 top-k clean action L2 | {_fmt(named_l2_mean)} |")
        lines.append(f"| multivector 平均 top-k clean action L2 | {_fmt(multivector_l2_mean)} |")
        lines.append("")
        lines.append("说明：L2 越小越好。")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    query_file = get_config_path(config, "query_file")
    db_data_dir = get_config_path(config, "db_data_dir")
    output_dir = ensure_output_dir(Path(config["output_dir"]) if "output_dir" in config else None)

    qdrant_cfg = config.get("qdrant", {})
    experiment_cfg = config.get("experiment", {})
    key_cfg = config.get("keys", {})
    if not isinstance(experiment_cfg, dict):
        raise ValueError("Config field 'experiment' must be an object")
    if not isinstance(key_cfg, dict):
        raise ValueError("Config field 'keys' must be an object")

    selected_keys = parse_selected_keys(key_cfg)
    weights = parse_weights(selected_keys, key_cfg)
    db_state = load_or_build_db_state(config, db_data_dir)
    db_stats = db_state.db_stats
    db_index = db_state.db_index

    mode = str(experiment_cfg.get("mode", "both"))
    if mode not in {"named", "multivector", "both"}:
        raise ValueError("experiment.mode must be one of: named, multivector, both")
    step_filter_mode = str(experiment_cfg.get("step_filter", "all"))
    if step_filter_mode not in {"exact", "window", "all"}:
        raise ValueError("experiment.step_filter must be one of: exact, window, all")
    step_window = int(experiment_cfg.get("step_window", 0))
    top_k = int(experiment_cfg.get("top_k", 10))
    candidate_limit = int(experiment_cfg.get("candidate_limit", 50))
    rrf_k = int(experiment_cfg.get("rrf_k", 60))
    max_query_steps = experiment_cfg.get("max_query_steps")
    query_step_idxs = experiment_cfg.get("query_step_idxs", [])
    if query_step_idxs is None:
        query_step_idxs = []
    if not isinstance(query_step_idxs, list):
        raise ValueError("experiment.query_step_idxs must be a list of ints")
    requested_step_idxs = {int(value) for value in query_step_idxs}

    named_collection = str(qdrant_cfg.get("named_collection", "openpi_steps_named"))
    multivector_collection = str(qdrant_cfg.get("multivector_collection", "openpi_steps_multivector"))
    prefer_http = bool(qdrant_cfg.get("prefer_http", False))

    query_step_names: list[str]
    with h5py.File(query_file.expanduser().resolve(), "r") as h5_file:
        query_step_names = sorted(key for key in h5_file.keys() if key.startswith("step_"))
    effective_query_steps = [
        name for name in query_step_names
        if not requested_step_idxs or int(name.split("_", maxsplit=1)[1]) in requested_step_idxs
    ]
    if max_query_steps is not None:
        effective_query_steps = effective_query_steps[:max_query_steps]

    log(f"Config file: {args.config}")
    log(f"Query file: {query_file}")
    log(f"DB data dir: {db_stats.data_dir}")
    log(f"DB steps: {db_stats.total_steps}")
    log(f"Query steps to run: {len(effective_query_steps)}")
    log(f"Selected keys: {selected_keys}")
    log(f"Normalized weights: {weights}")
    log(f"Mode: {mode}")
    log(f"Step filter: {step_filter_mode}")
    log(f"Candidate limit: {candidate_limit}, top_k: {top_k}, rrf_k: {rrf_k}")

    # Build query resources based on mode
    named_storage: CacheStorage | None = None
    client: QdrantClient | None = None

    if mode in {"named", "both"}:
        named_storage = make_named_storage(config, db_stats, selected_keys, weights)
        log(f"Named mode: CacheStorage ready (fields={list(named_storage._backend.vector_dims.keys())})")

    if mode in {"multivector", "both"}:
        ensure_rrf_support()
        client = make_client(config)
        version = client.info()
        log(f"Connected to Qdrant: title={version.title!r} version={version.version!r}")

    log(f"Transport: {'http' if prefer_http else 'grpc'}")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for query_index, query_record in enumerate(
        iter_query_records(
            query_file,
            db_stats,
            requested_step_idxs=requested_step_idxs,
            max_query_steps=max_query_steps,
        ),
        start=1,
    ):
        log(f"Running query step {query_index}/{len(effective_query_steps)}: {query_record.step_name}")
        # Qdrant filter for multivector mode
        query_filter = build_filter(step_filter_mode, query_record.step_idx, step_window)
        # Storage-compatible filter for named mode
        storage_query_filter = build_query_filter(step_filter_mode, query_record.step_idx, step_window)
        local_predicate = build_local_step_predicate(step_filter_mode, query_record.step_idx, step_window)
        background_clean_action_l2_mean = compute_background_clean_action_l2_mean(
            np.asarray(query_record.clean_action, dtype=np.float32).reshape(-1),
            db_index,
            local_predicate,
        )

        named_run: CollectionRun | None = None
        multivector_run: CollectionRun | None = None

        if mode in {"named", "both"}:
            named_run = run_named_collection(
                storage=named_storage,
                query_record=query_record,
                db_index=db_index,
                selected_keys=selected_keys,
                top_k=top_k,
                query_filter=storage_query_filter,
                step_filter_mode=step_filter_mode,
            )
        if mode in {"multivector", "both"}:
            multivector_run = run_collection(
                client=client,
                collection_name=multivector_collection,
                query_record=query_record,
                db_index=db_index,
                selected_keys=selected_keys,
                weights=weights,
                step_filter=query_filter,
                top_k=top_k,
                candidate_limit=candidate_limit,
                mode="multivector",
                db_stats=db_stats,
                rrf_k=rrf_k,
            )

        comparison = compare_runs(named_run, multivector_run)
        summary_row = {
            "query_file": str(query_file),
            "query_step_name": query_record.step_name,
            "query_step_idx": query_record.step_idx,
            "step_filter": step_filter_mode,
            "step_window": step_window,
            "selected_keys": json.dumps(selected_keys),
            "weights": json.dumps(weights, sort_keys=True),
            "background_clean_action_l2_mean": background_clean_action_l2_mean,
            "named_time_sec": named_run.total_time_sec if named_run else None,
            "multivector_time_sec": multivector_run.total_time_sec if multivector_run else None,
            "named_candidate_count": named_run.candidate_count if named_run else None,
            "multivector_candidate_count": multivector_run.candidate_count if multivector_run else None,
            "named_atomic_requests": named_run.atomic_request_count if named_run else None,
            "multivector_atomic_requests": multivector_run.atomic_request_count if multivector_run else None,
            "named_topk_ids": json.dumps([item.point_id for item in named_run.top_results]) if named_run else None,
            "multivector_topk_ids": json.dumps([item.point_id for item in multivector_run.top_results]) if multivector_run else None,
            "named_topk_clean_action_l2_mean": compute_topk_clean_action_l2_mean(named_run.top_results) if named_run else None,
            "multivector_topk_clean_action_l2_mean": (
                compute_topk_clean_action_l2_mean(multivector_run.top_results) if multivector_run else None
            ),
            **comparison,
        }
        summary_rows.append(summary_row)

        for label, run in (("named", named_run), ("multivector", multivector_run)):
            if run is None:
                continue
            for rank, result in enumerate(run.top_results, start=1):
                meta = db_index[result.point_id]
                detail_rows.append(
                    {
                        "query_file": str(query_file),
                        "query_step_name": query_record.step_name,
                        "query_step_idx": query_record.step_idx,
                        "collection_mode": label,
                        "collection_name": run.collection_name,
                        "rank": rank,
                        "point_id": result.point_id,
                        "source_file": meta.source_file,
                        "step_name": meta.step_name,
                        "step_idx": meta.step_idx,
                        "task": meta.task,
                        "episode_id": meta.episode_id,
                        "weighted_score": result.weighted_score,
                        "clean_action_l2": result.clean_action_l2,
                        "logical_scores": json.dumps(result.logical_scores, sort_keys=True),
                    }
                )

        log(
            f"{query_record.step_name}: "
            f"named_time={named_run.total_time_sec if named_run else None} "
            f"multi_time={multivector_run.total_time_sec if multivector_run else None} "
            f"bg_clean_l2={background_clean_action_l2_mean:.6f}"
        )

    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.csv"
    detail_path = output_dir / "details.csv"

    config_payload = {
        "query_file": str(query_file),
        "db_data_dir": str(db_data_dir),
        "mode": mode,
        "step_filter": step_filter_mode,
        "step_window": step_window,
        "selected_keys": selected_keys,
        "weights": weights,
        "top_k": top_k,
        "candidate_limit": candidate_limit,
        "rrf_k": rrf_k,
        "transport": "http" if prefer_http else "grpc",
        "named_collection": named_collection,
        "multivector_collection": multivector_collection,
        "source_config": config,
    }
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if detail_rows:
        with detail_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    report_path = write_markdown_report(
        output_dir=output_dir,
        config_payload=config_payload,
        summary_rows=summary_rows,
    )

    log(f"Saved config: {config_path}")
    log(f"Saved summary: {summary_path}")
    log(f"Saved details: {detail_path}")
    log(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
