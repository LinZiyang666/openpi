from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import models

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.qdrant_openpi_common import PAYLOAD_INDEXES
from exp.qdrant_openpi_common import build_multivector_vectors
from exp.qdrant_openpi_common import build_multivector_vectors_config
from exp.qdrant_openpi_common import build_named_vectors
from exp.qdrant_openpi_common import build_named_vectors_config
from exp.qdrant_openpi_common import build_payload
from exp.qdrant_openpi_common import dataset_summary_lines
from exp.qdrant_openpi_common import iter_step_locators
from exp.qdrant_openpi_common import iter_step_records
from exp.qdrant_openpi_common import load_step_record
from exp.qdrant_openpi_common import make_point_id
from exp.qdrant_openpi_common import scan_dataset
from exp.qdrant_openpi_common import StepLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest OpenPI HDF5 step data into Qdrant collections.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing episode .h5 files.")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant base URL.")
    parser.add_argument("--grpc-port", type=int, default=6334, help="Qdrant gRPC port.")
    parser.add_argument(
        "--mode",
        choices=("named", "multivector", "both"),
        default="both",
        help="Which collection(s) to populate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of points per upsert batch. Large step vectors usually require 1.",
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes for point preparation. Defaults to all logical CPUs.",
    )
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate target collections.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and convert data without touching Qdrant.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds for Qdrant HTTP operations.",
    )
    parser.add_argument(
        "--max-lang-tokens",
        type=int,
        default=None,
        help="Override named-vector padding length. Must be >= the observed dataset maximum.",
    )
    parser.add_argument("--named-collection", default="openpi_steps_named", help="Collection name for named vectors.")
    parser.add_argument(
        "--multivector-collection",
        default="openpi_steps_multivector",
        help="Collection name for multivectors.",
    )
    parser.add_argument(
        "--skip-payload-indexes",
        action="store_true",
        help="Do not create payload indexes on the target collections.",
    )
    parser.add_argument(
        "--enable-vector-indexing",
        action="store_true",
        help="Enable vector indexing. Disabled by default because these vectors are very large.",
    )
    parser.add_argument(
        "--prefer-http",
        action="store_true",
        help="Use HTTP instead of gRPC for requests. Not recommended for this workload.",
    )
    return parser.parse_args()


def _print_stats(prefix: str, lines: list[str]) -> None:
    print(prefix)
    for line in lines:
        print(f"  {line}")


def _make_client(args: argparse.Namespace) -> QdrantClient:
    return QdrantClient(
        url=args.url,
        grpc_port=args.grpc_port,
        prefer_grpc=not args.prefer_http,
        timeout=args.request_timeout,
    )


def _prepare_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vectors_config: dict[str, models.VectorParams],
    recreate: bool,
    create_payload_indexes: bool,
    request_timeout: int,
    enable_vector_indexing: bool,
) -> None:
    hnsw_config = None
    optimizers_config = None
    if not enable_vector_indexing:
        hnsw_config = models.HnswConfigDiff(m=0)
        optimizers_config = models.OptimizersConfigDiff(indexing_threshold=0)

    if recreate:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name=collection_name)
            print(f"Deleted collection: {collection_name}")

    if client.collection_exists(collection_name):
        print(f"Using existing collection: {collection_name}")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            on_disk_payload=True,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
        )
        print(f"Created collection: {collection_name}")

    if create_payload_indexes:
        for field_name, field_schema in PAYLOAD_INDEXES:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        print(f"Ensured payload indexes: {collection_name}")


def _flush_batch(
    client: QdrantClient,
    collection_name: str,
    batch: list[models.PointStruct],
    *,
    request_timeout: int,
) -> int:
    if not batch:
        return 0
    _ = request_timeout
    try:
        client.upsert(collection_name=collection_name, points=batch, wait=True)
    except UnexpectedResponse as exc:
        response_text = exc.content.decode("utf-8", errors="replace") if isinstance(exc.content, bytes) else str(exc)
        if "JSON payload" in response_text and "larger than allowed" in response_text:
            raise RuntimeError(
                "Qdrant rejected the upsert because the HTTP request body is too large. "
                "Retry with --batch-size 1. If the error still appears with batch-size 1, "
                "the next step is to switch this script to gRPC transport."
            ) from exc
        raise
    written = len(batch)
    batch.clear()
    return written


def _build_points_for_record(
    record,
    stats,
    mode: str,
) -> tuple[models.PointStruct | None, models.PointStruct | None]:
    point_id = make_point_id(record.source_file_display, record.step_name)
    payload = build_payload(
        record,
        max_lang_tokens=stats.max_lang_tokens,
        noise_action_fields=stats.schema.noise_action_fields,
    )

    named_point: models.PointStruct | None = None
    multivector_point: models.PointStruct | None = None
    if mode in {"named", "both"}:
        named_point = models.PointStruct(
            id=point_id,
            vector=build_named_vectors(record, stats),
            payload=payload,
        )
    if mode in {"multivector", "both"}:
        multivector_point = models.PointStruct(
            id=point_id,
            vector=build_multivector_vectors(record, stats),
            payload=payload,
        )
    return named_point, multivector_point


def _build_points_for_locator(
    locator: StepLocator,
    stats,
    mode: str,
) -> tuple[models.PointStruct | None, models.PointStruct | None]:
    return _build_points_for_record(load_step_record(locator, stats), stats, mode)


def _await_upsert_future(
    future: Future[tuple[str, int]] | None,
    totals: dict[str, int],
) -> None:
    if future is None:
        return
    collection_name, written = future.result()
    totals[collection_name] = totals.get(collection_name, 0) + written


def _submit_collection_batch(
    *,
    executor: ThreadPoolExecutor | None,
    pending_future: Future[tuple[str, int]] | None,
    client: QdrantClient,
    collection_name: str,
    batch: list[models.PointStruct],
    request_timeout: int,
    totals: dict[str, int],
) -> Future[tuple[str, int]] | None:
    if not batch:
        return pending_future
    _await_upsert_future(pending_future, totals)
    points = list(batch)
    batch.clear()
    if executor is None:
        written = _flush_batch(client, collection_name, points, request_timeout=request_timeout)
        totals[collection_name] = totals.get(collection_name, 0) + written
        return None
    return executor.submit(_flush_batch_with_name, client, collection_name, points, request_timeout)


def _flush_batch_with_name(
    client: QdrantClient,
    collection_name: str,
    batch: list[models.PointStruct],
    request_timeout: int,
) -> tuple[str, int]:
    written = _flush_batch(client, collection_name, batch, request_timeout=request_timeout)
    return collection_name, written


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.prepare_workers <= 0:
        raise ValueError("--prepare-workers must be >= 1")

    stats = scan_dataset(args.data_dir, requested_max_lang_tokens=args.max_lang_tokens)
    _print_stats("Dataset scan complete:", dataset_summary_lines(stats))

    if args.dry_run:
        converted = 0
        for record in iter_step_records(stats):
            if args.mode in {"named", "both"}:
                build_named_vectors(record, stats)
            if args.mode in {"multivector", "both"}:
                build_multivector_vectors(record, stats)
            converted += 1
        print(f"Dry run succeeded: converted {converted} step records without writing to Qdrant.")
        return

    admin_client = _make_client(args)
    version = admin_client.info()
    print(f"Connected to Qdrant: title={version.title!r} version={version.version!r}")
    print(f"Transport: {'http' if args.prefer_http else 'grpc'}")
    if args.enable_vector_indexing:
        print("Vector indexing: enabled")
    else:
        print("Vector indexing: disabled (plain/exact search mode)")
    print(f"Prepare workers: {args.prepare_workers}")

    if args.mode in {"named", "both"}:
        _prepare_collection(
            admin_client,
            collection_name=args.named_collection,
            vectors_config=build_named_vectors_config(stats),
            recreate=args.recreate,
            create_payload_indexes=not args.skip_payload_indexes,
            request_timeout=args.request_timeout,
            enable_vector_indexing=args.enable_vector_indexing,
        )
    if args.mode in {"multivector", "both"}:
        _prepare_collection(
            admin_client,
            collection_name=args.multivector_collection,
            vectors_config=build_multivector_vectors_config(stats),
            recreate=args.recreate,
            create_payload_indexes=not args.skip_payload_indexes,
            request_timeout=args.request_timeout,
            enable_vector_indexing=args.enable_vector_indexing,
        )

    named_client = _make_client(args) if args.mode in {"named", "both"} else None
    multivector_client = _make_client(args) if args.mode in {"multivector", "both"} else None
    upsert_executor = ThreadPoolExecutor(max_workers=2 if args.mode == "both" else 1)
    prepare_executor = ProcessPoolExecutor(max_workers=args.prepare_workers) if args.prepare_workers > 1 else None
    named_batch: list[models.PointStruct] = []
    multivector_batch: list[models.PointStruct] = []
    totals: dict[str, int] = {
        args.named_collection: 0,
        args.multivector_collection: 0,
    }
    named_future: Future[tuple[str, int]] | None = None
    multivector_future: Future[tuple[str, int]] | None = None
    prepared_queue: deque[tuple[int, Future[tuple[models.PointStruct | None, models.PointStruct | None]]]] = deque()

    def process_prepared(index: int, prepared: tuple[models.PointStruct | None, models.PointStruct | None]) -> None:
        nonlocal named_future, multivector_future
        named_point, multivector_point = prepared
        if named_point is not None:
            named_batch.append(named_point)
            if len(named_batch) >= args.batch_size:
                named_future = _submit_collection_batch(
                    executor=upsert_executor,
                    pending_future=named_future,
                    client=named_client,
                    collection_name=args.named_collection,
                    batch=named_batch,
                    request_timeout=args.request_timeout,
                    totals=totals,
                )
        if multivector_point is not None:
            multivector_batch.append(multivector_point)
            if len(multivector_batch) >= args.batch_size:
                multivector_future = _submit_collection_batch(
                    executor=upsert_executor,
                    pending_future=multivector_future,
                    client=multivector_client,
                    collection_name=args.multivector_collection,
                    batch=multivector_batch,
                    request_timeout=args.request_timeout,
                    totals=totals,
                )
        if index % 25 == 0 or index == stats.total_steps:
            print(f"Processed {index}/{stats.total_steps} steps")

    try:
        step_iter = iter_step_locators(stats) if prepare_executor is not None else iter_step_records(stats)
        for index, item in enumerate(step_iter, start=1):
            if prepare_executor is None:
                process_prepared(index, _build_points_for_record(item, stats, args.mode))
                continue

            prepared_queue.append(
                (
                    index,
                    prepare_executor.submit(_build_points_for_locator, item, stats, args.mode),
                )
            )
            while len(prepared_queue) >= args.prepare_workers:
                ready_index, future = prepared_queue.popleft()
                process_prepared(ready_index, future.result())

        while prepared_queue:
            ready_index, future = prepared_queue.popleft()
            process_prepared(ready_index, future.result())

        if args.mode in {"named", "both"}:
            named_future = _submit_collection_batch(
                executor=upsert_executor,
                pending_future=named_future,
                client=named_client,
                collection_name=args.named_collection,
                batch=named_batch,
                request_timeout=args.request_timeout,
                totals=totals,
            )
        if args.mode in {"multivector", "both"}:
            multivector_future = _submit_collection_batch(
                executor=upsert_executor,
                pending_future=multivector_future,
                client=multivector_client,
                collection_name=args.multivector_collection,
                batch=multivector_batch,
                request_timeout=args.request_timeout,
                totals=totals,
            )

        _await_upsert_future(named_future, totals)
        _await_upsert_future(multivector_future, totals)
    finally:
        if prepare_executor is not None:
            prepare_executor.shutdown(wait=True)
        upsert_executor.shutdown(wait=True)

    if args.mode in {"named", "both"}:
        named_count = named_client.count(collection_name=args.named_collection, exact=True).count
        print(f"Named collection written={totals[args.named_collection]} stored_count={named_count}")
    if args.mode in {"multivector", "both"}:
        multivector_count = multivector_client.count(collection_name=args.multivector_collection, exact=True).count
        print(
            f"Multivector collection written={totals[args.multivector_collection]} stored_count={multivector_count}"
        )


if __name__ == "__main__":
    main()
