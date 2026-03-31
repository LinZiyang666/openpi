from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.qdrant_openpi_common import CLEAN_ACTION_FIELD
from exp.qdrant_openpi_common import NamedVectorChunk
from exp.qdrant_openpi_common import PROMPT_FIELD
from exp.qdrant_openpi_common import VISION_FIELDS
from exp.qdrant_openpi_common import build_multivector_vectors
from exp.qdrant_openpi_common import build_named_vectors
from exp.qdrant_openpi_common import dataset_summary_lines
from exp.qdrant_openpi_common import make_point_id
from exp.qdrant_openpi_common import named_vector_chunks_map
from exp.qdrant_openpi_common import scan_dataset
from exp.qdrant_openpi_common import select_sample_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify OpenPI Qdrant ingestion and run self-query checks.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing episode .h5 files.")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant base URL.")
    parser.add_argument("--grpc-port", type=int, default=6334, help="Qdrant gRPC port.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds for Qdrant HTTP operations.",
    )
    parser.add_argument("--named-collection", default="openpi_steps_named", help="Collection name for named vectors.")
    parser.add_argument(
        "--multivector-collection",
        default="openpi_steps_multivector",
        help="Collection name for multivectors.",
    )
    parser.add_argument(
        "--max-lang-tokens",
        type=int,
        default=None,
        help="Override named-vector padding length. Must be >= the observed dataset maximum.",
    )
    parser.add_argument(
        "--step-idx",
        type=int,
        default=None,
        help="Optional payload filter for self-query checks. Also selects the sample record when possible.",
    )
    parser.add_argument("--limit", type=int, default=5, help="KNN limit for each verification query.")
    parser.add_argument(
        "--prefer-http",
        action="store_true",
        help="Use HTTP instead of gRPC for requests.",
    )
    return parser.parse_args()


def _maybe_step_filter(step_idx: int | None) -> models.Filter | None:
    if step_idx is None:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="step_idx",
                match=models.MatchValue(value=step_idx),
            )
        ]
    )


def _print_stats(lines: list[str]) -> None:
    print("Dataset scan complete:")
    for line in lines:
        print(f"  {line}")


def _print_query_result(label: str, response: models.QueryResponse, expected_id: int) -> None:
    if not response.points:
        raise RuntimeError(f"{label}: query returned no points")
    top = response.points[0]
    actual_id = int(top.id)
    is_match = actual_id == expected_id
    payload = top.payload or {}
    print(
        f"{label}: top1_id={actual_id} expected_id={expected_id} "
        f"score={top.score:.6f} self_match={is_match} "
        f"source_file={payload.get('source_file')} step_name={payload.get('step_name')}"
    )


def _query_named_field(
    client: QdrantClient,
    *,
    collection_name: str,
    field_name: str,
    vector_map: dict[str, list[float]],
    chunks: tuple[NamedVectorChunk, ...],
    query_filter: models.Filter | None,
    limit: int,
    timeout: int,
) -> models.QueryResponse:
    _ = field_name
    _ = timeout
    if len(chunks) == 1:
        return client.query_points(
            collection_name=collection_name,
            query=vector_map[chunks[0].vector_name],
            using=chunks[0].vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=["source_file", "step_name", "step_idx"],
        )

    prefetch = [
        models.Prefetch(
            query=vector_map[chunk.vector_name],
            using=chunk.vector_name,
            filter=query_filter,
            limit=limit,
        )
        for chunk in chunks
    ]
    return client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=["source_file", "step_name", "step_idx"],
    )


def main() -> None:
    args = parse_args()
    stats = scan_dataset(args.data_dir, requested_max_lang_tokens=args.max_lang_tokens)
    _print_stats(dataset_summary_lines(stats))

    client = QdrantClient(
        url=args.url,
        grpc_port=args.grpc_port,
        prefer_grpc=not args.prefer_http,
        timeout=args.request_timeout,
    )
    version = client.info()
    print(f"Connected to Qdrant: title={version.title!r} version={version.version!r}")
    print(f"Transport: {'http' if args.prefer_http else 'grpc'}")

    named_count = client.count(collection_name=args.named_collection, exact=True).count
    multivector_count = client.count(collection_name=args.multivector_collection, exact=True).count
    print(f"named_count={named_count} expected_steps={stats.total_steps}")
    print(f"multivector_count={multivector_count} expected_steps={stats.total_steps}")

    sample = select_sample_record(stats, step_idx=args.step_idx)
    expected_id = make_point_id(sample.source_file_display, sample.step_name)
    query_filter = _maybe_step_filter(args.step_idx)
    print(
        f"Sample record: source_file={sample.source_file_display} step_name={sample.step_name} "
        f"step_idx={sample.step_idx} point_id={expected_id}"
    )

    named_vectors = build_named_vectors(sample, stats)
    named_chunks = named_vector_chunks_map(stats)
    for field_name in (VISION_FIELDS[0], PROMPT_FIELD, CLEAN_ACTION_FIELD):
        response = _query_named_field(
            client,
            collection_name=args.named_collection,
            field_name=field_name,
            vector_map=named_vectors,
            chunks=named_chunks[field_name],
            query_filter=query_filter,
            limit=args.limit,
            timeout=args.request_timeout,
        )
        _print_query_result(f"named/{field_name}", response, expected_id)

    multivector_vectors = build_multivector_vectors(sample, stats)
    for field_name in (PROMPT_FIELD,):
        response = client.query_points(
            collection_name=args.multivector_collection,
            query=multivector_vectors[field_name],
            using=field_name,
            query_filter=query_filter,
            limit=args.limit,
            with_payload=["source_file", "step_name", "step_idx"],
        )
        _print_query_result(f"multivector/{field_name}", response, expected_id)


if __name__ == "__main__":
    main()
