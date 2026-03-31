# Qdrant Experiment Plan

## Goal

Build a Qdrant-based retrieval experiment for collected OpenPI HDF5 episodes under a directory such as `data/libero_spatial_1`.

The experiment will use one Qdrant instance:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
```

Inside that instance, we will create two separate collections:

1. `openpi_steps_named`
2. `openpi_steps_multivector`

They are not two separate databases. They are two collections inside the same Qdrant service.

## Source Data Model

The current collector writes one HDF5 file per episode. Each episode file contains:

- top-level attributes:
  - `experiment_name`
  - `task`
  - `episode_id`
  - `num_steps`
  - `timestamp`
  - `success`
- top-level groups:
  - `step_0000`
  - `step_0001`
  - ...

Each `step_xxxx` group is a retrieval unit and contains:

- `vision_0`: `float16[256, 2048]`
- `vision_1`: `float16[256, 2048]`
- `vision_2`: `float16[256, 2048]`
- `prompt_emb`: `float16[num_lang_tokens, 2048]`
- `robot_state`: `float32[32]`
- `noise_action_1 ... noise_action_9`: `float32[action_horizon, action_dim]`
- `clean_action`: `float32[action_horizon, action_dim]`

For the current `pi05_libero` data, typical shapes are:

- `prompt_emb`: `(200, 2048)`
- `robot_state`: `(32,)`
- `noise_action_*`: `(10, 32)`
- `clean_action`: `(10, 32)`

## Retrieval Granularity

Each point in Qdrant will correspond to one `step_xxxx`, not one whole episode.

This is the right granularity because:

- each step already contains a full set of queryable tensors
- step-level retrieval is more flexible than episode-level retrieval
- payload filtering can still recover episode-level context

## Collection 1: Standard Named Vectors

### Name

`openpi_steps_named`

### Why this collection exists

This collection is the baseline.

Qdrant standard dense vectors require a single fixed-length vector per named field. Therefore, matrix-shaped tensors such as `vision_0 (256, 2048)` cannot be inserted directly as ordinary dense vectors. They must first be reduced to one fixed-length vector.

### Vector schema

Each point will store named vectors:

- `vision_0`: `524288`
- `vision_1`: `524288`
- `vision_2`: `524288`
- `prompt_emb`: `MAX_LANG_TOKENS * 2048`
- `robot_state`: `32`
- `noise_action_1`: `320`
- `noise_action_2`: `320`
- `noise_action_3`: `320`
- `noise_action_4`: `320`
- `noise_action_5`: `320`
- `noise_action_6`: `320`
- `noise_action_7`: `320`
- `noise_action_8`: `320`
- `noise_action_9`: `320`
- `clean_action`: `320`

### Tensor-to-vector conversion rules

- `vision_0/1/2`: flatten `(256, 2048)` to `(524288,)`
- `prompt_emb`: pad from `(num_lang_tokens, 2048)` to `(MAX_LANG_TOKENS, 2048)`, then flatten
- `robot_state`: keep as `(32,)`
- `noise_action_*`: flatten `(10, 32)` to `(320,)`
- `clean_action`: flatten `(10, 32)` to `(320,)`

No pooling and no dimensionality reduction are allowed in this collection. The goal is full value preservation.

### Prompt padding rule

`prompt_emb` is variable-length across samples, but Qdrant standard dense vectors require a fixed vector size for the same named field.

Therefore:

- scan the dataset first to compute `MAX_LANG_TOKENS`
- right-pad shorter `prompt_emb` arrays with zeros to `(MAX_LANG_TOKENS, 2048)`
- store the original token length in payload as `num_lang_tokens`

This keeps the representation lossless with respect to the original values. The only added values are explicit padding zeros, which can be removed after retrieval using the stored original length.

### Distance defaults

- `vision_*`: `Cosine`
- `prompt_emb`: `Cosine`
- `robot_state`: `Euclid`
- `noise_action_*`: `Euclid`
- `clean_action`: `Euclid`

### Tradeoff

This schema preserves all original values, but it no longer preserves the original 2D structure directly inside the vector field. Shape reconstruction requires:

- the stored payload shape metadata
- reshape after retrieval

Storage cost is also much higher than a pooled baseline.

## Collection 2: Multivector

### Name

`openpi_steps_multivector`

### Why this collection exists

This collection preserves the internal matrix structure for tensors that are naturally a sequence of vectors.

Qdrant multivector supports storing a variable number of same-shaped dense vectors in a single point. This fits:

- vision token embeddings
- prompt token embeddings
- action horizon sequences

Qdrant documentation states multivectors are available as of `v1.10.0`.

### Vector schema

This collection will use named vectors. Some names are ordinary dense vectors, some are multivectors.

- `vision_0`: multivector, each row size `2048`
- `vision_1`: multivector, each row size `2048`
- `vision_2`: multivector, each row size `2048`
- `prompt_emb`: multivector, each row size `2048`
- `robot_state`: dense vector, size `32`
- `noise_action_1`: multivector, each row size `32`
- `noise_action_2`: multivector, each row size `32`
- `noise_action_3`: multivector, each row size `32`
- `noise_action_4`: multivector, each row size `32`
- `noise_action_5`: multivector, each row size `32`
- `noise_action_6`: multivector, each row size `32`
- `noise_action_7`: multivector, each row size `32`
- `noise_action_8`: multivector, each row size `32`
- `noise_action_9`: multivector, each row size `32`
- `clean_action`: multivector, each row size `32`

### Multivector comparator

Use:

- `MultiVectorComparator.MAX_SIM`

This is the default Qdrant late-interaction style comparator and is the most reasonable starting point for the experiment.

### Distance defaults

- `vision_*`: `Cosine`
- `prompt_emb`: `Cosine`
- `robot_state`: `Euclid`
- `noise_action_*`: `Euclid`
- `clean_action`: `Euclid`

### Tradeoff

This schema preserves the original sequence structure directly, with no flattening and no pooling, but collection creation and querying are more complex and storage cost is higher.

## Payload Schema

Every point in both collections should carry the same payload fields:

- `experiment_name`
- `task`
- `episode_id`
- `step_idx`
- `step_name`
- `success`
- `timestamp`
- `source_file`
- `num_lang_tokens`
- `max_lang_tokens`
- `vision_shape`
- `prompt_shape`
- `robot_state_shape`
- `action_shape`

Recommended payload values:

- `source_file`: full or relative HDF5 path
- `step_name`: for example `step_0007`
- `step_idx`: integer parsed from `step_0007`
- `vision_shape`: for example `[256, 2048]`
- `prompt_shape`: for example `[200, 2048]`
- `action_shape`: for example `[10, 32]`
- `max_lang_tokens`: dataset-wide padding length used in `openpi_steps_named`

## Point ID Strategy

Do not use only `episode_id + step_idx`.

Reason:

- current sample files under `data/libero_spatial_1` contain repeated `episode_id=0`
- IDs must remain unique across files

Use a stable hash generated from:

```text
source_file + "::" + step_name
```

Recommended implementation:

- generate a deterministic 64-bit integer hash
- use the same ID for both collections

This keeps re-imports idempotent and avoids accidental collisions from repeated episode counters.

## Import Pipeline

Implement a script, for example:

```bash
python scripts/qdrant_ingest_openpi.py --data-dir data/libero_spatial_1 --mode both
```

### Script responsibilities

1. connect to `http://localhost:6333`
2. check that the server is reachable
3. check Qdrant version if possible
4. create or recreate the target collections
5. scan all `.h5` files under the input directory
6. read episode attributes
7. iterate through all `step_xxxx`
8. convert tensors according to the chosen collection schema
9. build payload
10. batch upsert points into Qdrant

For `openpi_steps_named`, the script must perform a first pass over the dataset to compute the global `MAX_LANG_TOKENS` before collection creation and insertion.

### Suggested CLI

- `--data-dir`
- `--url`
- `--mode named|multivector|both`
- `--batch-size`
- `--recreate`
- `--named-collection`
- `--multivector-collection`
- `--max-lang-tokens`

If `--max-lang-tokens` is omitted, the script should infer it from the dataset automatically.

## Query Validation

After ingestion, run a minimal validation script.

### Required checks

1. point count matches the total number of `step_xxxx` groups across all files
2. fetch one known step and query by its own vector
3. top-1 result should usually return the same point
4. test at least these fields:
   - `vision_0`
   - `prompt_emb`
   - `clean_action`

### Example validation targets

- self-query in `openpi_steps_named` using flattened `vision_0`
- self-query in `openpi_steps_named` using padded-and-flattened `prompt_emb`
- self-query in `openpi_steps_named` using flattened `clean_action`
- self-query in `openpi_steps_multivector` using raw `prompt_emb`

## Implementation Order

1. implement reusable HDF5 step iterator
2. implement stable point ID function
3. implement payload builder
4. implement first-pass scan for `MAX_LANG_TOKENS`
5. implement tensor conversion for named-vector collection with flatten and prompt padding
6. implement tensor conversion for multivector collection with raw matrix preservation
7. implement collection creation helpers
8. implement batched upsert
9. implement simple query verification

## Practical Notes

- `qdrant-client` is available locally in the current environment
- the service at `http://localhost:6333` still needs runtime reachability validation from the actual machine environment
- multivector support requires a sufficiently new Qdrant server
- both collections should be created as idempotently as possible for repeated experiments
- the updated plan requires preserving all values from the original tensors; no data reduction should be used in either collection

## Experiment Config File

The step-level retrieval experiment is driven by a JSON config file. Example:

```bash
python -m exp.qdrant_step_knn_experiment --config exp/qdrant_step_knn_experiment_config.example.json
```

The example config file is:

- `exp/qdrant_step_knn_experiment_config.example.json`

### Top-Level Fields

#### `query_file`

Path to the single HDF5 file used as the query source.

Example:

```json
"query_file": "data/libero_spatial_1/episode_0000_20260331_035016_586551.h5"
```

The script will iterate over all `step_xxxx` groups inside this file unless further restricted.

#### `db_data_dir`

Path to the HDF5 directory corresponding to the database corpus. This is used for:

- dataset scanning
- building the local metadata cache
- background `clean_action` evaluation

Example:

```json
"db_data_dir": "data/libero_spatial"
```

#### `output_dir`

Directory where the experiment outputs are written.

The script writes:

- `config.json`
- `summary.csv`
- `details.csv`
- `report.md`

Example:

```json
"output_dir": "exp_outputs/qdrant_step_knn_example"
```

### `cache`

Controls the local cache used for:

- dataset scan results
- point metadata
- local `clean_action` evaluation support

#### `cache.enabled`

- `true`: use cache
- `false`: always rebuild from the raw HDF5 files

Example:

```json
"enabled": true
```

#### `cache.dir`

Directory where cache files are stored.

Example:

```json
"dir": ".cache/qdrant_step_knn"
```

#### `cache.refresh`

- `false`: reuse cache if valid
- `true`: force cache rebuild

This is useful when:

- the dataset changed
- cache format changed
- evaluation logic changed and you want a clean rebuild

Example:

```json
"refresh": false
```

### `qdrant`

Connection and collection settings.

#### `qdrant.url`

Base HTTP URL of the Qdrant service.

Example:

```json
"url": "http://localhost:6333"
```

#### `qdrant.grpc_port`

gRPC port used by the client when `prefer_http=false`.

Example:

```json
"grpc_port": 6334
```

#### `qdrant.prefer_http`

- `false`: prefer gRPC
- `true`: use HTTP

For this workload, gRPC is usually the better choice.

Example:

```json
"prefer_http": false
```

#### `qdrant.request_timeout`

Request timeout in seconds.

Example:

```json
"request_timeout": 1800
```

Use a large timeout for heavy multivector retrieval experiments.

#### `qdrant.named_collection`

Name of the standard named-vector collection.

Example:

```json
"named_collection": "openpi_steps_named"
```

#### `qdrant.multivector_collection`

Name of the multivector collection.

Example:

```json
"multivector_collection": "openpi_steps_multivector"
```

### `experiment`

Controls retrieval behavior.

#### `experiment.mode`

Controls which collection(s) to query.

Allowed values:

- `"named"`
- `"multivector"`
- `"both"`

Meaning:

- `"named"`: only query `openpi_steps_named`
- `"multivector"`: only query `openpi_steps_multivector`
- `"both"`: query both and compare the results

Example:

```json
"mode": "named"
```

#### `experiment.step_filter`

Controls whether retrieval is restricted by `step_idx`.

Allowed values:

- `"all"`
- `"exact"`
- `"window"`

Meaning:

- `"all"`: search the whole collection
- `"exact"`: only search points with `step_idx == query_step_idx`
- `"window"`: only search points with `abs(step_idx - query_step_idx) <= step_window`

Examples:

```json
"step_filter": "all"
```

```json
"step_filter": "exact"
```

```json
"step_filter": "window"
```

#### `experiment.step_window`

Only used when `step_filter="window"`.

Example:

```json
"step_window": 5
```

This means:

```text
abs(candidate_step_idx - query_step_idx) <= 5
```

For example, if the query step is `step_0007`, then candidate step indices from `2` to `12` are allowed.

When `step_filter` is `"all"` or `"exact"`, this field is ignored.

#### `experiment.top_k`

How many final retrieval results to keep per query step.

Example:

```json
"top_k": 10
```

This affects:

- final `summary.csv`
- final `details.csv`
- `report.md`

#### `experiment.candidate_limit`

How many candidates each prefetch branch may return before fusion.

Example:

```json
"candidate_limit": 50
```

Larger values:

- may improve recall
- increase query cost

#### `experiment.rrf_k`

The `k` parameter used in weighted RRF fusion.

Example:

```json
"rrf_k": 60
```

This controls how strongly top-ranked items dominate the fused ranking.

#### `experiment.query_step_idxs`

Restrict the experiment to a fixed subset of query steps from the query HDF5 file.

Example:

```json
"query_step_idxs": [0, 5, 9]
```

Meaning:

- only run `step_0000`
- `step_0005`
- `step_0009`

Use an empty list to allow all query steps:

```json
"query_step_idxs": []
```

#### `experiment.max_query_steps`

Optional cap on how many query steps are actually executed.

Example:

```json
"max_query_steps": 3
```

Meaning:

- after all filters are applied, only run the first 3 query steps

Use:

```json
"max_query_steps": null
```

to disable the cap.

### `keys`

Controls which logical retrieval keys participate in the database query and with what weights.

Each key supports two forms:

#### Boolean form

Example:

```json
"robot_state": false
```

Meaning:

- `false`: disable the key
- `true`: enable the key with default weight `1.0`

#### Object form

Example:

```json
"vision_0": {
  "enabled": true,
  "weight": 1.0
}
```

Meaning:

- `enabled`: whether to use this key
- `weight`: the raw weight before normalization

The script normalizes all enabled weights automatically so their sum is `1`.

#### Supported keys

- `vision_0`
- `vision_1`
- `vision_2`
- `prompt_emb`
- `robot_state`
- `noise_action_1`
- `noise_action_2`
- `noise_action_3`
- `noise_action_4`
- `noise_action_5`
- `noise_action_6`
- `noise_action_7`
- `noise_action_8`
- `noise_action_9`
- `clean_action`

#### Example key block

```json
"keys": {
  "vision_0": {
    "enabled": true,
    "weight": 1.0
  },
  "vision_1": {
    "enabled": false,
    "weight": 2.0
  },
  "prompt_emb": {
    "enabled": true,
    "weight": 0.5
  },
  "robot_state": {
    "enabled": true,
    "weight": 10.0
  },
  "clean_action": false
}
```

This means:

- query by `vision_0`
- query by `prompt_emb`
- query by `robot_state`
- ignore `vision_1`
- ignore `clean_action`

The raw weights `1.0`, `0.5`, and `10.0` will be normalized internally before retrieval.

### Notes on Metric Choice

In the current experiment script, the query metric used by each field is determined by the collection schema already stored in Qdrant.

That means:

- `vision_*` and `prompt_emb` use the metric defined at collection creation time
- `robot_state`, `noise_action_*`, and `clean_action` also use the metric defined at collection creation time

For the current collections, this is typically:

- `vision_*`: `Cosine`
- `prompt_emb`: `Cosine`
- `robot_state`: `Euclid`
- `noise_action_*`: `Euclid`
- `clean_action`: `Euclid`

If you want a different metric for a field, you must rebuild the collection with a different `VectorParams.distance` setting. The query config file cannot override the distance dynamically after the collection has already been created.

## References

- Qdrant vectors documentation: <https://qdrant.tech/documentation/concepts/vectors/>
- Qdrant collections documentation: <https://qdrant.tech/documentation/concepts/collections/>
