"""Toy Qdrant query server.

Receives vision/prompt/robot_state embeddings from ``toy_stage1_server.py``
over HTTP+msgpack, queries Qdrant, and returns the best-matching
``clean_action``.

This script runs **outside** the uv environment (qdrant_client is
incompatible with the project's uv lockfile).  It depends only on:

    qdrant_client, numpy, h5py, flask, msgpack, msgpack_numpy

All query configuration (keys, weights, mode, step_filter, RRF, etc.)
uses the same JSON config format and the same functions as
``qdrant_step_knn_experiment.py``.

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

# Ensure repo root is on sys.path so ``exp.*`` imports work when invoked as
# ``python exp/toy_qdrant_server.py``.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from exp.qdrant_openpi_common import (
    StepRecord,
    build_multivector_vectors,
    build_named_vectors,
    named_vector_chunks_map,
    scan_dataset,
)
from exp.qdrant_step_knn_experiment import (
    DBPointMeta,
    build_db_index,
    build_filter,
    build_prefetches_for_collection,
    ensure_rrf_support,
    load_config,
    load_or_build_db_state,
    make_client,
    parse_selected_keys,
    parse_weights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_QUERY_KEYS = frozenset({
    "vision_0",
    "vision_1",
    "vision_2",
    "prompt_emb",
    "robot_state",
})

# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------


class ToyQdrantQueryServer:
    """Stateful query handler backed by a Qdrant collection.

    Configuration parsing and query construction reuse the same functions as
    ``qdrant_step_knn_experiment.py``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        db_stats: Any,
        db_index: dict[int, DBPointMeta],
        client: QdrantClient,
    ) -> None:
        self.config = config
        self.db_stats = db_stats
        self.db_index = db_index
        self.client = client

        # ---- Validate and parse config ----
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
        self.named_collection: str = str(qdrant_cfg.get("named_collection", "openpi_steps_named"))
        self.multivector_collection: str = str(
            qdrant_cfg.get("multivector_collection", "openpi_steps_multivector")
        )

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

        for key_name, value in key_cfg.items():
            if key_name in ALLOWED_QUERY_KEYS:
                continue
            enabled = False
            if isinstance(value, bool):
                enabled = value
            elif isinstance(value, dict):
                enabled = bool(value.get("enabled", False))
            if enabled:
                raise ValueError(
                    f"Key '{key_name}' is enabled in config but not available in "
                    f"toy mode. Only {sorted(ALLOWED_QUERY_KEYS)} are supported."
                )

    # ---- Query ----

    def query(
        self,
        vision_0: np.ndarray,
        vision_1: np.ndarray,
        vision_2: np.ndarray,
        prompt_emb: np.ndarray,
        robot_state: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        """Query Qdrant and return the best-matching clean_action.

        Returns:
            clean_action as float32 array with shape ``clean_action_shape``.
        """
        # Build a synthetic StepRecord for the query vector builders
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

        # Build step filter (from qdrant_step_knn_experiment.build_filter)
        step_filter = build_filter(self.step_filter_mode, step_idx, self.step_window)

        # Build prefetches (from qdrant_step_knn_experiment.build_prefetches_for_collection)
        prefetches, fusion_weights, atomic_count = build_prefetches_for_collection(
            selected_keys=self.selected_keys,
            weights=self.weights,
            query_record=record,
            step_filter=step_filter,
            candidate_limit=self.candidate_limit,
            mode=self.mode,
            db_stats=self.db_stats,
        )

        # Choose collection
        collection_name = (
            self.named_collection if self.mode == "named" else self.multivector_collection
        )

        def _query_points(active_filter: models.Filter | None):
            active_prefetches, active_fusion_weights, _ = build_prefetches_for_collection(
                selected_keys=self.selected_keys,
                weights=self.weights,
                query_record=record,
                step_filter=active_filter,
                candidate_limit=self.candidate_limit,
                mode=self.mode,
                db_stats=self.db_stats,
            )
            return self.client.query_points(
                collection_name=collection_name,
                prefetch=active_prefetches,
                query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k, weights=active_fusion_weights)),
                limit=self.top_k,
                with_payload=False,
                with_vectors=False,
            )

        # Query Qdrant with RRF fusion.
        response = _query_points(step_filter)
        if not response.points and self.step_filter_mode != "all":
            logger.warning(
                "No Qdrant results for step_idx=%d on collection=%s with step_filter=%s; retrying without step filter",
                step_idx,
                collection_name,
                self.step_filter_mode,
            )
            response = _query_points(None)

        if not response.points:
            raise RuntimeError(
                f"Qdrant returned no results for step_idx={step_idx} on "
                f"collection={collection_name}, even after fallback"
            )

        # Look up clean_action from local index
        point_id = int(response.points[0].id)
        if point_id not in self.db_index:
            raise RuntimeError(
                f"Point ID {point_id} returned by Qdrant not found in local DB index. "
                "The index may be stale or incomplete."
            )
        meta = self.db_index[point_id]
        clean_action = meta.clean_action_flat.reshape(self.db_stats.schema.clean_action_shape)
        return clean_action.astype(np.float32)


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

    # Validate RRF support
    ensure_rrf_support()

    # Load DB data and build local index
    db_data_dir = Path(str(config.get("db_data_dir", "")))
    if not db_data_dir.as_posix() or db_data_dir.as_posix() == ".":
        raise ValueError("Config must include a non-empty 'db_data_dir' field.")

    logger.info("Loading DB state from %s ...", db_data_dir)
    db_state = load_or_build_db_state(config, db_data_dir)
    db_stats = db_state.db_stats
    db_index = db_state.db_index
    logger.info("DB index ready: %d points", len(db_index))

    # Create Qdrant client
    client = make_client(config)
    version_info = client.info()
    logger.info("Connected to Qdrant: %s %s", version_info.title, version_info.version)

    # Create server instance
    server = ToyQdrantQueryServer(config, db_stats, db_index, client)

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
