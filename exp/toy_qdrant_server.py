"""Toy Qdrant query server.

Receives vision/prompt/robot_state embeddings from ``toy_stage1_server.py``
over HTTP+msgpack, queries Qdrant, and returns the best-matching
``clean_action``.

This script runs **outside** the uv environment (qdrant_client is
incompatible with the project's uv lockfile).  It depends only on:

    qdrant_client, numpy, h5py, flask, msgpack, msgpack_numpy

Query modes
-----------
named (default)
    Uses CacheStorage (QdrantVectorStore) as the query entry point.
    Fusion strategy (RRF weights, rrf_k, candidate_limit) is declared in
    QdrantBackendConfig and handled entirely inside the backend.
    Only fields supported by the storage interface are used; noise_action_*
    and clean_action are silently ignored if selected.

multivector
    Unchanged: direct QdrantClient with server-side RRF.  Multivector
    collections are Qdrant-specific and outside the storage abstraction.

clean_action retrieval
----------------------
clean_action is returned from the local db_index (in-memory) rather than
from the Qdrant payload.  This avoids large tensor transfer for every query.
When data is re-ingested in CacheEntry format, switch to fetch_payload().

Usage
-----
::

    python exp/toy_qdrant_server.py \\
        --config exp/toy_qdrant_config.json \\
        --port 8100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Imports from sibling experiment modules
# ---------------------------------------------------------------------------

import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from exp.qdrant_openpi_common import (
    StepRecord,
    build_multivector_vectors,
    named_vector_chunks_map,
    scan_dataset,
)
from exp.qdrant_step_knn_experiment import (
    DBPointMeta,
    build_db_index,
    build_prefetches_for_collection,
    ensure_rrf_support,
    load_config,
    load_or_build_db_state,
    make_client,
    parse_selected_keys,
    parse_weights,
)

# ---------------------------------------------------------------------------
# Storage layer imports (named mode)
# ---------------------------------------------------------------------------

from openpi.cache.types import VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE
from openpi.cache.storage_types import QueryFilter, QuerySpec
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.backends.qdrant_backend import QdrantVectorStore, QdrantBackendConfig

logger = logging.getLogger(__name__)

# Fields that the storage interface handles (superset check in query)
_STORAGE_FIELD_INPUTS = (VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_vector_dims(db_stats, selected_keys: list[str]) -> dict[str, int]:
    """Derive per-field full flat dims from db_stats.

    The backend handles Qdrant's 65 535-dim chunk limit internally, so all
    field sizes are valid regardless of how many chunks they require.
    """
    chunk_map = named_vector_chunks_map(db_stats)
    return {
        field_name: sum(c.end - c.start for c in chunks)
        for field_name in selected_keys
        if (chunks := chunk_map.get(field_name))
    }


def _make_named_storage(
    config: dict[str, Any],
    db_stats,
    selected_keys: list[str],
    weights: dict[str, float],
) -> CacheStorage:
    qdrant_cfg = config.get("qdrant", {})
    experiment_cfg = config.get("experiment", {})

    vector_dims = _derive_vector_dims(db_stats, selected_keys)
    if not vector_dims:
        raise RuntimeError(
            "No recognised fields found for storage interface. "
            "Check that selected_keys contains valid CACHE_QUERY_FIELDS entries."
        )

    top_k = int(experiment_cfg.get("top_k", 1))
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


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------


class ToyQdrantQueryServer:
    """Stateful query handler backed by a Qdrant collection.

    Named mode  : uses CacheStorage.search() — backend handles fusion.
    Multivector : uses QdrantClient directly — unchanged.
    """

    def __init__(
        self,
        config: dict[str, Any],
        db_stats: Any,
        db_index: dict[int, DBPointMeta],
        storage_or_client: CacheStorage | QdrantClient,
    ) -> None:
        self.config = config
        self.db_stats = db_stats
        self.db_index = db_index

        self._validate_config()

        experiment_cfg = config.get("experiment", {})
        qdrant_cfg = config.get("qdrant", {})
        key_cfg = config.get("keys", {})

        self.mode: str = str(experiment_cfg.get("mode", "named"))
        self.selected_keys = parse_selected_keys(key_cfg)
        self.weights = parse_weights(self.selected_keys, key_cfg)
        self.step_filter_mode: str = str(experiment_cfg.get("step_filter", "all"))
        self.step_window: int = int(experiment_cfg.get("step_window", 0))
        self.top_k: int = int(experiment_cfg.get("top_k", 1))
        self.candidate_limit: int = int(experiment_cfg.get("candidate_limit", 50))
        self.rrf_k: int = int(experiment_cfg.get("rrf_k", 60))
        self.multivector_collection: str = str(
            qdrant_cfg.get("multivector_collection", "openpi_steps_multivector")
        )

        if self.mode == "named":
            self._storage: CacheStorage = storage_or_client
            self._client: QdrantClient | None = None
        else:
            self._storage = None
            self._client = storage_or_client

        logger.info("Mode: %s", self.mode)
        logger.info("Selected keys: %s", self.selected_keys)
        logger.info("Normalized weights: %s", self.weights)
        logger.info("Step filter: %s (window=%d)", self.step_filter_mode, self.step_window)
        logger.info("top_k=%d, candidate_limit=%d, rrf_k=%d", self.top_k, self.candidate_limit, self.rrf_k)

    # ---- Validation ----

    def _validate_config(self) -> None:
        experiment_cfg = self.config.get("experiment", {})
        if not isinstance(experiment_cfg, dict):
            raise ValueError("Config field 'experiment' must be an object")

        mode = str(experiment_cfg.get("mode", "named"))
        if mode == "both":
            raise ValueError(
                "toy_qdrant_server does not support mode='both'. "
                "Use 'named' or 'multivector'."
            )
        if mode not in ("named", "multivector"):
            raise ValueError(f"Unsupported mode: {mode}")

        step_filter = str(experiment_cfg.get("step_filter", "all"))
        if step_filter not in ("exact", "window", "all"):
            raise ValueError(f"experiment.step_filter must be one of: exact, window, all (got {step_filter})")

        key_cfg = self.config.get("keys", {})
        if not isinstance(key_cfg, dict):
            raise ValueError("Config field 'keys' must be an object")

        _ALLOWED = frozenset({VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE})
        if mode == "multivector":
            for key_name, value in key_cfg.items():
                if key_name in _ALLOWED:
                    continue
                enabled = False
                if isinstance(value, bool):
                    enabled = value
                elif isinstance(value, dict):
                    enabled = bool(value.get("enabled", False))
                if enabled:
                    raise ValueError(
                        f"Key '{key_name}' is enabled in config but not available in "
                        f"toy mode. Only {sorted(_ALLOWED)} are supported."
                    )

    # ---- Query entry point ----

    def query(
        self,
        vision_0: np.ndarray,
        vision_1: np.ndarray,
        vision_2: np.ndarray,
        prompt_emb: np.ndarray,
        robot_state: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        """Query and return the best-matching clean_action as float32 array."""
        if self.mode == "named":
            return self._query_via_storage(vision_0, vision_1, vision_2, prompt_emb, robot_state, step_idx)
        else:
            return self._query_via_client(vision_0, vision_1, vision_2, prompt_emb, robot_state, step_idx)

    # ---- Named mode: CacheStorage path ----

    def _query_via_storage(
        self,
        vision_0: np.ndarray,
        vision_1: np.ndarray,
        vision_2: np.ndarray,
        prompt_emb: np.ndarray,
        robot_state: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        _inputs: dict[str, np.ndarray] = {
            VISION_0: vision_0,
            VISION_1: vision_1,
            VISION_2: vision_2,
            PROMPT_EMB: prompt_emb,
            ROBOT_STATE: robot_state,
        }
        # Only include fields that are both selected and supported by the backend
        supported = self._storage._backend.vector_dims
        query_keys = {
            field: np.asarray(arr, dtype=np.float32).reshape(-1)
            for field, arr in _inputs.items()
            if field in self.selected_keys and field in supported
        }
        if not query_keys:
            raise RuntimeError(
                "No selected fields are supported by the storage backend. "
                f"Selected: {self.selected_keys}, Backend fields: {set(supported.keys())}"
            )

        spec = QuerySpec(
            query_keys=query_keys,
            top_k=self.top_k,
            filters=self._make_query_filter(step_idx),
        )

        results = self._storage.search(spec)
        if not results and self.step_filter_mode != "all":
            import dataclasses
            logger.warning(
                "No results for step_idx=%d with step_filter=%s; retrying without filter",
                step_idx,
                self.step_filter_mode,
            )
            results = self._storage.search(dataclasses.replace(spec, filters=None))

        if not results:
            raise RuntimeError(
                f"Storage returned no results for step_idx={step_idx}, even after fallback"
            )

        return self._clean_action_from_index(int(results[0].id))

    def _make_query_filter(self, step_idx: int) -> QueryFilter | None:
        if self.step_filter_mode == "all":
            return None
        if self.step_filter_mode == "exact":
            return QueryFilter(step_range=(step_idx, step_idx))
        if self.step_filter_mode == "window":
            return QueryFilter(step_range=(step_idx - self.step_window, step_idx + self.step_window))
        return None

    # ---- Multivector mode: direct QdrantClient path (unchanged) ----

    def _query_via_client(
        self,
        vision_0: np.ndarray,
        vision_1: np.ndarray,
        vision_2: np.ndarray,
        prompt_emb: np.ndarray,
        robot_state: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        record = StepRecord(
            source_file=Path("__live_query__"),
            source_file_display="__live_query__",
            experiment_name="",
            task="",
            episode_id=0,
            success=False,
            timestamp="",
            num_episode_steps=0,
            step_name=f"step_{step_idx:04d}",
            step_idx=step_idx,
            vision={
                "vision_0": np.asarray(vision_0),
                "vision_1": np.asarray(vision_1),
                "vision_2": np.asarray(vision_2),
            },
            prompt_emb=np.asarray(prompt_emb),
            robot_state=np.asarray(robot_state),
            clean_action=np.zeros(self.db_stats.schema.clean_action_shape, dtype=np.float32),
            noise_actions={},
        )

        def _query_points(active_filter):
            active_prefetches, active_fusion_weights, _ = build_prefetches_for_collection(
                selected_keys=self.selected_keys,
                weights=self.weights,
                query_record=record,
                step_filter=active_filter,
                candidate_limit=self.candidate_limit,
                mode="multivector",
                db_stats=self.db_stats,
            )
            return self._client.query_points(
                collection_name=self.multivector_collection,
                prefetch=active_prefetches,
                query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k, weights=active_fusion_weights)),
                limit=self.top_k,
                with_payload=False,
                with_vectors=False,
            )

        from exp.qdrant_step_knn_experiment import build_filter
        step_filter = build_filter(self.step_filter_mode, step_idx, self.step_window)
        response = _query_points(step_filter)
        if not response.points and self.step_filter_mode != "all":
            logger.warning(
                "No results for step_idx=%d with step_filter=%s; retrying without filter",
                step_idx,
                self.step_filter_mode,
            )
            response = _query_points(None)

        if not response.points:
            raise RuntimeError(
                f"Qdrant returned no results for step_idx={step_idx}, even after fallback"
            )

        return self._clean_action_from_index(int(response.points[0].id))

    # ---- Shared: local index lookup ----

    def _clean_action_from_index(self, point_id: int) -> np.ndarray:
        if point_id not in self.db_index:
            raise RuntimeError(
                f"Point ID {point_id} returned by Qdrant not found in local DB index. "
                "The index may be stale or incomplete."
            )
        meta = self.db_index[point_id]
        return meta.clean_action_flat.reshape(self.db_stats.schema.clean_action_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy Qdrant query server for toy_stage1_server.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config JSON file.")
    parser.add_argument("--port", type=int, default=8100, help="HTTP port to listen on.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host to bind to.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_cfg = config.get("experiment", {})
    mode = str(experiment_cfg.get("mode", "named"))

    if mode == "multivector":
        ensure_rrf_support()

    db_data_dir = Path(str(config.get("db_data_dir", "")))
    if not db_data_dir.as_posix() or db_data_dir.as_posix() == ".":
        raise ValueError("Config must include a non-empty 'db_data_dir' field.")

    logger.info("Loading DB state from %s ...", db_data_dir)
    db_state = load_or_build_db_state(config, db_data_dir)
    db_stats = db_state.db_stats
    db_index = db_state.db_index
    logger.info("DB index ready: %d points", len(db_index))

    key_cfg = config.get("keys", {})
    selected_keys = parse_selected_keys(key_cfg)
    weights = parse_weights(selected_keys, key_cfg)

    if mode == "named":
        storage_or_client = _make_named_storage(config, db_stats, selected_keys, weights)
        logger.info("Named mode: using CacheStorage (QdrantVectorStore)")
    else:
        storage_or_client = make_client(config)
        version_info = storage_or_client.info()
        logger.info("Multivector mode: direct QdrantClient %s %s", version_info.title, version_info.version)

    server = ToyQdrantQueryServer(config, db_stats, db_index, storage_or_client)

    # ---- Flask app ----
    import msgpack
    import msgpack_numpy
    from flask import Flask, Response, request

    msgpack_numpy.patch()

    app = Flask(__name__)

    @app.route("/query", methods=["POST"])
    def handle_query():
        payload = msgpack.unpackb(request.data, object_hook=msgpack_numpy.decode)
        try:
            clean_action = server.query(
                vision_0=payload["vision_0"],
                vision_1=payload["vision_1"],
                vision_2=payload["vision_2"],
                prompt_emb=payload["prompt_emb"],
                robot_state=payload["robot_state"],
                step_idx=int(payload["step_idx"]),
            )
        except Exception as exc:
            logger.exception("Query failed")
            error_body = msgpack.packb({"error": str(exc)}, default=msgpack_numpy.encode)
            return Response(error_body, status=500, content_type="application/x-msgpack")

        result = {"clean_action": clean_action.astype(np.float32)}
        return Response(
            msgpack.packb(result, default=msgpack_numpy.encode),
            content_type="application/x-msgpack",
        )

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return "OK\n", 200

    logger.info("Starting Flask server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
