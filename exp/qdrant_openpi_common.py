from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
from qdrant_client import models

VISION_FIELDS = ("vision_0", "vision_1", "vision_2")
PROMPT_FIELD = "prompt_emb"
ROBOT_STATE_FIELD = "robot_state"
CLEAN_ACTION_FIELD = "clean_action"
NOISE_ACTION_RE = re.compile(r"^noise_action_(\d+)$")
MAX_DENSE_VECTOR_DIM = 65_535


@dataclass(frozen=True)
class DatasetSchema:
    vision_shapes: dict[str, tuple[int, int]]
    prompt_width: int
    robot_state_shape: tuple[int, ...]
    clean_action_shape: tuple[int, int]
    noise_action_fields: tuple[str, ...]
    noise_action_shapes: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class DatasetStats:
    data_dir: Path
    h5_paths: tuple[Path, ...]
    schema: DatasetSchema
    total_steps: int
    max_lang_tokens: int


@dataclass(frozen=True)
class StepRecord:
    source_file: Path
    source_file_display: str
    experiment_name: str
    task: str
    episode_id: int
    success: bool
    timestamp: str
    num_episode_steps: int
    step_name: str
    step_idx: int
    vision: dict[str, np.ndarray]
    prompt_emb: np.ndarray
    robot_state: np.ndarray
    clean_action: np.ndarray
    noise_actions: dict[str, np.ndarray]


@dataclass(frozen=True)
class StepLocator:
    source_file: Path
    step_name: str


@dataclass(frozen=True)
class NamedVectorChunk:
    vector_name: str
    start: int
    end: int


def resolve_h5_paths(data_dir: str | Path) -> tuple[Path, ...]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    paths = tuple(sorted(root.rglob("*.h5")))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found under: {root}")
    return paths


def sort_noise_action_fields(keys: Any) -> tuple[str, ...]:
    matched: list[tuple[int, str]] = []
    for key in keys:
        found = NOISE_ACTION_RE.match(key)
        if found:
            matched.append((int(found.group(1)), key))
    return tuple(name for _, name in sorted(matched))


def sorted_step_names(keys: Any) -> tuple[str, ...]:
    return tuple(sorted(key for key in keys if key.startswith("step_")))


def _shape_of(group: h5py.Group, field_name: str) -> tuple[int, ...]:
    return tuple(int(dim) for dim in group[field_name].shape)


def _schema_from_group(group: h5py.Group) -> DatasetSchema:
    present_fields = set(group.keys())
    missing = [field for field in (*VISION_FIELDS, PROMPT_FIELD, ROBOT_STATE_FIELD, CLEAN_ACTION_FIELD) if field not in present_fields]
    if missing:
        raise ValueError(f"Missing required fields in {group.name}: {missing}")

    noise_action_fields = sort_noise_action_fields(group.keys())
    if not noise_action_fields:
        raise ValueError(f"No noise_action_* fields found in {group.name}")

    vision_shapes = {field: _shape_of(group, field) for field in VISION_FIELDS}
    for field, shape in vision_shapes.items():
        if len(shape) != 2:
            raise ValueError(f"{group.name}/{field} must be 2D, got shape={shape}")

    prompt_shape = _shape_of(group, PROMPT_FIELD)
    if len(prompt_shape) != 2:
        raise ValueError(f"{group.name}/{PROMPT_FIELD} must be 2D, got shape={prompt_shape}")

    robot_state_shape = _shape_of(group, ROBOT_STATE_FIELD)
    if len(robot_state_shape) != 1:
        raise ValueError(f"{group.name}/{ROBOT_STATE_FIELD} must be 1D, got shape={robot_state_shape}")

    clean_action_shape = _shape_of(group, CLEAN_ACTION_FIELD)
    if len(clean_action_shape) != 2:
        raise ValueError(f"{group.name}/{CLEAN_ACTION_FIELD} must be 2D, got shape={clean_action_shape}")

    noise_action_shapes = {field: _shape_of(group, field) for field in noise_action_fields}
    for field, shape in noise_action_shapes.items():
        if len(shape) != 2:
            raise ValueError(f"{group.name}/{field} must be 2D, got shape={shape}")

    return DatasetSchema(
        vision_shapes=vision_shapes,
        prompt_width=prompt_shape[1],
        robot_state_shape=robot_state_shape,
        clean_action_shape=clean_action_shape,
        noise_action_fields=noise_action_fields,
        noise_action_shapes=noise_action_shapes,
    )


def _validate_group_schema(group: h5py.Group, schema: DatasetSchema) -> int:
    found_schema = _schema_from_group(group)
    if found_schema.vision_shapes != schema.vision_shapes:
        raise ValueError(f"Vision shapes mismatch in {group.name}: {found_schema.vision_shapes} != {schema.vision_shapes}")
    if found_schema.prompt_width != schema.prompt_width:
        raise ValueError(f"Prompt width mismatch in {group.name}: {found_schema.prompt_width} != {schema.prompt_width}")
    if found_schema.robot_state_shape != schema.robot_state_shape:
        raise ValueError(
            f"robot_state shape mismatch in {group.name}: {found_schema.robot_state_shape} != {schema.robot_state_shape}"
        )
    if found_schema.clean_action_shape != schema.clean_action_shape:
        raise ValueError(
            f"clean_action shape mismatch in {group.name}: {found_schema.clean_action_shape} != {schema.clean_action_shape}"
        )
    if found_schema.noise_action_fields != schema.noise_action_fields:
        raise ValueError(
            f"noise_action fields mismatch in {group.name}: {found_schema.noise_action_fields} != {schema.noise_action_fields}"
        )
    if found_schema.noise_action_shapes != schema.noise_action_shapes:
        raise ValueError(
            f"noise_action shapes mismatch in {group.name}: {found_schema.noise_action_shapes} != {schema.noise_action_shapes}"
        )
    return int(group[PROMPT_FIELD].shape[0])


def scan_dataset(data_dir: str | Path, *, requested_max_lang_tokens: int | None = None) -> DatasetStats:
    h5_paths = resolve_h5_paths(data_dir)
    data_root = Path(data_dir).expanduser().resolve()
    schema: DatasetSchema | None = None
    total_steps = 0
    observed_max_lang_tokens = 0

    for path in h5_paths:
        with h5py.File(path, "r") as h5_file:
            step_names = sorted_step_names(h5_file.keys())
            if not step_names:
                raise ValueError(f"No step_* groups found in file: {path}")
            for step_name in step_names:
                group = h5_file[step_name]
                if schema is None:
                    schema = _schema_from_group(group)
                    observed_max_lang_tokens = int(group[PROMPT_FIELD].shape[0])
                else:
                    observed_max_lang_tokens = max(observed_max_lang_tokens, _validate_group_schema(group, schema))
                total_steps += 1

    if schema is None:
        raise ValueError(f"No step data found under: {data_root}")

    max_lang_tokens = observed_max_lang_tokens if requested_max_lang_tokens is None else requested_max_lang_tokens
    if max_lang_tokens < observed_max_lang_tokens:
        raise ValueError(
            f"Requested max_lang_tokens={max_lang_tokens} is smaller than observed maximum {observed_max_lang_tokens}"
        )

    return DatasetStats(
        data_dir=data_root,
        h5_paths=h5_paths,
        schema=schema,
        total_steps=total_steps,
        max_lang_tokens=max_lang_tokens,
    )


def dataset_summary_lines(stats: DatasetStats) -> list[str]:
    lines = [
        f"data_dir={stats.data_dir}",
        f"h5_files={len(stats.h5_paths)}",
        f"total_steps={stats.total_steps}",
        f"max_lang_tokens={stats.max_lang_tokens}",
    ]
    for field in VISION_FIELDS:
        lines.append(f"{field}_shape={stats.schema.vision_shapes[field]}")
    lines.append(f"prompt_width={stats.schema.prompt_width}")
    lines.append(f"robot_state_shape={stats.schema.robot_state_shape}")
    lines.append(f"clean_action_shape={stats.schema.clean_action_shape}")
    lines.append(f"noise_action_fields={stats.schema.noise_action_fields}")
    return lines


def _display_path(path: Path) -> str:
    cwd = Path.cwd().resolve()
    try:
        return path.resolve().relative_to(cwd).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_step_record(locator: StepLocator, stats: DatasetStats) -> StepRecord:
    path = locator.source_file.resolve()
    with h5py.File(path, "r") as h5_file:
        attrs = dict(h5_file.attrs)
        group = h5_file[locator.step_name]
        vision = {field: np.asarray(group[field]) for field in VISION_FIELDS}
        prompt_emb = np.asarray(group[PROMPT_FIELD])
        robot_state = np.asarray(group[ROBOT_STATE_FIELD])
        clean_action = np.asarray(group[CLEAN_ACTION_FIELD])
        noise_actions = {field: np.asarray(group[field]) for field in stats.schema.noise_action_fields}
        return StepRecord(
            source_file=path,
            source_file_display=_display_path(path),
            experiment_name=str(attrs["experiment_name"]),
            task=str(attrs["task"]),
            episode_id=int(attrs["episode_id"]),
            success=bool(attrs["success"]),
            timestamp=str(attrs["timestamp"]),
            num_episode_steps=int(attrs["num_steps"]),
            step_name=locator.step_name,
            step_idx=int(locator.step_name.split("_", maxsplit=1)[1]),
            vision=vision,
            prompt_emb=prompt_emb,
            robot_state=robot_state,
            clean_action=clean_action,
            noise_actions=noise_actions,
        )


def iter_step_locators(stats: DatasetStats) -> Any:
    for path in stats.h5_paths:
        with h5py.File(path, "r") as h5_file:
            for step_name in sorted_step_names(h5_file.keys()):
                yield StepLocator(source_file=path.resolve(), step_name=step_name)


def iter_step_records(stats: DatasetStats) -> Any:
    for locator in iter_step_locators(stats):
        yield load_step_record(locator, stats)


def select_sample_record(stats: DatasetStats, *, step_idx: int | None = None) -> StepRecord:
    for record in iter_step_records(stats):
        if step_idx is None or record.step_idx == step_idx:
            return record
    if step_idx is None:
        raise ValueError("No step records available")
    raise ValueError(f"No step record found with step_idx={step_idx}")


def make_point_id(source_file_display: str, step_name: str) -> int:
    key = f"{source_file_display}::{step_name}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def build_payload(record: StepRecord, *, max_lang_tokens: int, noise_action_fields: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "point_key": f"{record.source_file_display}::{record.step_name}",
        "source_file": record.source_file_display,
        "experiment_name": record.experiment_name,
        "task": record.task,
        "episode_id": record.episode_id,
        "num_episode_steps": record.num_episode_steps,
        "step_name": record.step_name,
        "step_idx": record.step_idx,
        "success": record.success,
        "timestamp": record.timestamp,
        "num_lang_tokens": int(record.prompt_emb.shape[0]),
        "max_lang_tokens": int(max_lang_tokens),
        "vision_0_shape": list(record.vision["vision_0"].shape),
        "vision_1_shape": list(record.vision["vision_1"].shape),
        "vision_2_shape": list(record.vision["vision_2"].shape),
        "prompt_shape": list(record.prompt_emb.shape),
        "robot_state_shape": list(record.robot_state.shape),
        "clean_action_shape": list(record.clean_action.shape),
        "noise_action_fields": list(noise_action_fields),
        "vision_dtype": str(record.vision["vision_0"].dtype),
        "prompt_dtype": str(record.prompt_emb.dtype),
        "robot_state_dtype": str(record.robot_state.dtype),
        "action_dtype": str(record.clean_action.dtype),
    }
    if noise_action_fields:
        payload["noise_action_shape"] = list(record.noise_actions[noise_action_fields[0]].shape)
    return payload


def _chunk_vector_name(field_name: str, chunk_index: int, total_chunks: int) -> str:
    if total_chunks == 1:
        return field_name
    return f"{field_name}__chunk_{chunk_index:03d}"


def chunk_layout_for_size(field_name: str, total_size: int, *, max_chunk_dim: int = MAX_DENSE_VECTOR_DIM) -> tuple[NamedVectorChunk, ...]:
    if total_size <= 0:
        raise ValueError(f"total_size must be positive for {field_name}, got {total_size}")
    chunks: list[NamedVectorChunk] = []
    start = 0
    chunk_index = 0
    while start < total_size:
        end = min(start + max_chunk_dim, total_size)
        chunks.append(NamedVectorChunk(vector_name="", start=start, end=end))
        start = end
        chunk_index += 1
    total_chunks = len(chunks)
    return tuple(
        NamedVectorChunk(
            vector_name=_chunk_vector_name(field_name, chunk_index=index, total_chunks=total_chunks),
            start=chunk.start,
            end=chunk.end,
        )
        for index, chunk in enumerate(chunks)
    )


def named_vector_chunks_map(stats: DatasetStats) -> dict[str, tuple[NamedVectorChunk, ...]]:
    chunk_map: dict[str, tuple[NamedVectorChunk, ...]] = {}
    for field in VISION_FIELDS:
        rows, width = stats.schema.vision_shapes[field]
        chunk_map[field] = chunk_layout_for_size(field, rows * width)
    chunk_map[PROMPT_FIELD] = chunk_layout_for_size(field_name=PROMPT_FIELD, total_size=stats.max_lang_tokens * stats.schema.prompt_width)
    chunk_map[ROBOT_STATE_FIELD] = chunk_layout_for_size(field_name=ROBOT_STATE_FIELD, total_size=stats.schema.robot_state_shape[0])
    clean_rows, clean_width = stats.schema.clean_action_shape
    chunk_map[CLEAN_ACTION_FIELD] = chunk_layout_for_size(field_name=CLEAN_ACTION_FIELD, total_size=clean_rows * clean_width)
    for field in stats.schema.noise_action_fields:
        rows, width = stats.schema.noise_action_shapes[field]
        chunk_map[field] = chunk_layout_for_size(field_name=field, total_size=rows * width)
    return chunk_map


def flatten_matrix(array: np.ndarray) -> list[float]:
    return np.asarray(array).reshape(-1).tolist()


def dense_vector(array: np.ndarray) -> list[float]:
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D dense vector, got shape={arr.shape}")
    return arr.tolist()


def pad_and_flatten_prompt(prompt_emb: np.ndarray, *, max_lang_tokens: int, prompt_width: int) -> list[float]:
    prompt = np.asarray(prompt_emb)
    if prompt.ndim != 2:
        raise ValueError(f"prompt_emb must be 2D, got shape={prompt.shape}")
    if prompt.shape[1] != prompt_width:
        raise ValueError(f"prompt_emb width mismatch: {prompt.shape[1]} != {prompt_width}")
    if prompt.shape[0] > max_lang_tokens:
        raise ValueError(f"prompt_emb length {prompt.shape[0]} exceeds max_lang_tokens={max_lang_tokens}")
    padded = np.zeros((max_lang_tokens, prompt_width), dtype=prompt.dtype)
    padded[: prompt.shape[0], :] = prompt
    return padded.reshape(-1).tolist()


def matrix_to_multivector(array: np.ndarray, *, width: int | None = None) -> list[list[float]]:
    matrix = np.asarray(array)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D multivector matrix, got shape={matrix.shape}")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"Multivector width mismatch: {matrix.shape[1]} != {width}")
    return matrix.tolist()


def build_named_vectors(record: StepRecord, stats: DatasetStats) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    flattened: dict[str, list[float]] = {field: flatten_matrix(record.vision[field]) for field in VISION_FIELDS}
    flattened[PROMPT_FIELD] = pad_and_flatten_prompt(
        record.prompt_emb,
        max_lang_tokens=stats.max_lang_tokens,
        prompt_width=stats.schema.prompt_width,
    )
    flattened[ROBOT_STATE_FIELD] = dense_vector(record.robot_state)
    flattened[CLEAN_ACTION_FIELD] = flatten_matrix(record.clean_action)
    for field in stats.schema.noise_action_fields:
        flattened[field] = flatten_matrix(record.noise_actions[field])

    for field_name, chunks in named_vector_chunks_map(stats).items():
        source = flattened[field_name]
        for chunk in chunks:
            vectors[chunk.vector_name] = source[chunk.start : chunk.end]
    return vectors


def build_multivector_vectors(record: StepRecord, stats: DatasetStats) -> dict[str, list[float] | list[list[float]]]:
    vectors: dict[str, list[float] | list[list[float]]] = {
        field: matrix_to_multivector(record.vision[field], width=stats.schema.vision_shapes[field][1]) for field in VISION_FIELDS
    }
    vectors[PROMPT_FIELD] = matrix_to_multivector(record.prompt_emb, width=stats.schema.prompt_width)
    vectors[ROBOT_STATE_FIELD] = dense_vector(record.robot_state)
    vectors[CLEAN_ACTION_FIELD] = matrix_to_multivector(record.clean_action, width=stats.schema.clean_action_shape[1])
    for field in stats.schema.noise_action_fields:
        vectors[field] = matrix_to_multivector(record.noise_actions[field], width=stats.schema.noise_action_shapes[field][1])
    return vectors


def build_named_vectors_config(stats: DatasetStats) -> dict[str, models.VectorParams]:
    config: dict[str, models.VectorParams] = {}
    distances: dict[str, models.Distance] = {
        **{field: models.Distance.COSINE for field in VISION_FIELDS},
        PROMPT_FIELD: models.Distance.COSINE,
        ROBOT_STATE_FIELD: models.Distance.EUCLID,
        CLEAN_ACTION_FIELD: models.Distance.EUCLID,
        **{field: models.Distance.EUCLID for field in stats.schema.noise_action_fields},
    }
    for field_name, chunks in named_vector_chunks_map(stats).items():
        for chunk in chunks:
            config[chunk.vector_name] = models.VectorParams(
                size=chunk.end - chunk.start,
                distance=distances[field_name],
                datatype=(
                    models.Datatype.FLOAT16
                    if field_name in (*VISION_FIELDS, PROMPT_FIELD)
                    else models.Datatype.FLOAT32
                ),
            )
    return config


def build_multivector_vectors_config(stats: DatasetStats) -> dict[str, models.VectorParams]:
    multivector = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
    config: dict[str, models.VectorParams] = {}
    for field in VISION_FIELDS:
        config[field] = models.VectorParams(
            size=stats.schema.vision_shapes[field][1],
            distance=models.Distance.COSINE,
            datatype=models.Datatype.FLOAT16,
            multivector_config=multivector,
        )
    config[PROMPT_FIELD] = models.VectorParams(
        size=stats.schema.prompt_width,
        distance=models.Distance.COSINE,
        datatype=models.Datatype.FLOAT16,
        multivector_config=multivector,
    )
    config[ROBOT_STATE_FIELD] = models.VectorParams(
        size=stats.schema.robot_state_shape[0],
        distance=models.Distance.EUCLID,
        datatype=models.Datatype.FLOAT32,
    )
    config[CLEAN_ACTION_FIELD] = models.VectorParams(
        size=stats.schema.clean_action_shape[1],
        distance=models.Distance.EUCLID,
        datatype=models.Datatype.FLOAT32,
        multivector_config=multivector,
    )
    for field in stats.schema.noise_action_fields:
        config[field] = models.VectorParams(
            size=stats.schema.noise_action_shapes[field][1],
            distance=models.Distance.EUCLID,
            datatype=models.Datatype.FLOAT32,
            multivector_config=multivector,
        )
    return config


PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("source_file", models.PayloadSchemaType.KEYWORD),
    ("experiment_name", models.PayloadSchemaType.KEYWORD),
    ("task", models.PayloadSchemaType.KEYWORD),
    ("episode_id", models.PayloadSchemaType.INTEGER),
    ("step_name", models.PayloadSchemaType.KEYWORD),
    ("step_idx", models.PayloadSchemaType.INTEGER),
    ("success", models.PayloadSchemaType.BOOL),
)
