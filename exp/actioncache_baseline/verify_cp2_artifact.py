"""Verify a CP2 library against its CP1 source (plan §3.7 (a)–(g), fail-closed).

Checks, each of which aborts the verification on first violation:

  (a)  entry ``id`` set == source set (no missing, no duplicates)
  (a') every entry carries ``checkpoint_id == CP2`` (the backend filters on it;
       a single CP1-tagged entry would silently vanish from every search)
  (a'') a real ``InMemoryBackend`` loads the artifact and a CP2 query with an
       entry's own key returns that entry as top-1
  (b)  payload tensors bit-identical to the source; trajectory_id / step_idx /
       outcome / prev_ids / next_ids identical
  (c)  every prev/next edge points inside the artifact
  (d)  ``vlm_out`` keys are finite float32 vectors of length ``d``
  (e)  ``action_chunk`` shapes agree with the source consensus and every entry
       carries the ``start_t=0.1`` intermediate (N_hit=1 tier)
  (f)  ``vector_dims == {"vlm_out": d}``
  (g)  binding metadata present (projection, id_policy, source sha, H5
       manifest, model, tokenizer) and the projection digest re-derives from
       the recorded (seed, d, p, D)

Usage:
  uv run python -m exp.actioncache_baseline.verify_cp2_artifact \\
      --cp2-pkl <cp2>.pkl --source-pkl <cp1>.pkl [--search-samples 20]
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch

from exp.actioncache_baseline import libs
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec
from openpi.cache.storage_types import QuerySpec
from openpi.cache.types import CheckpointID


class VerificationError(AssertionError):
    """A verifier check failed."""


def _as_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _payload_equal(a, b) -> bool:
    if not np.array_equal(_as_np(a.action_chunk), _as_np(b.action_chunk)):
        return False
    ia, ib = a.intermediates or {}, b.intermediates or {}
    if set(ia) != set(ib):
        return False
    for t in ia:
        if not np.array_equal(_as_np(ia[t]), _as_np(ib[t])):
            return False
    return a.denoising_num_steps == b.denoising_num_steps and a.task_key == b.task_key


def cp2_query_spec(key: torch.Tensor, top_k: int = 1) -> QuerySpec:
    """The deployed CP2 retrieval spec (single cosine field, affine_clip to [0, 1])."""
    return QuerySpec(
        query_keys={libs.FIELD: key},
        top_k=top_k,
        checkpoint_id=CheckpointID.CP2,
        fusion_weights={libs.FIELD: 1.0},
        fusion_method="weighted_score_sum",
        field_similarity={libs.FIELD: {"type": "cosine"}},
        score_normalization={"type": "per_field", "fields": {
            libs.FIELD: {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}}}},
    )


def verify(cp2_path: str, source_path: str, *, search_samples: int = 20, seed: int = 0) -> dict:
    art = libs.load_pickle(cp2_path)
    src = libs.load_pickle(source_path)
    entries = list(art["entries"])
    src_by_id = {e.id: e for e in src["entries"]}
    if len(src_by_id) != len(src["entries"]):
        raise VerificationError("source library has duplicate ids")

    # (a)
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise VerificationError("(a) duplicate ids in CP2 artifact")
    if set(ids) != set(src_by_id):
        missing = sorted(set(src_by_id) - set(ids))[:5]
        extra = sorted(set(ids) - set(src_by_id))[:5]
        raise VerificationError(f"(a) id set differs from source: missing={missing} extra={extra}")

    # (f) / (g) metadata
    d = int(art["vector_dims"].get(libs.FIELD, -1)) if isinstance(art.get("vector_dims"), dict) else -1
    if art.get("vector_dims") != {libs.FIELD: d} or d < 1:
        raise VerificationError(f"(f) vector_dims must be {{'{libs.FIELD}': d}}, got {art.get('vector_dims')}")
    if art.get("key_builder_type") != libs.KEY_BUILDER_TYPE:
        raise VerificationError(f"(g) key_builder_type {art.get('key_builder_type')!r}")
    if art.get("id_policy") != libs.ID_POLICY:
        raise VerificationError(f"(g) id_policy {art.get('id_policy')!r}")
    proj = art.get("projection")
    if not isinstance(proj, dict):
        raise VerificationError("(g) projection metadata missing")
    expected = get_projection_spec(proj["seed"], proj["d"], proj["p"], proj["D"]).meta()
    if expected != proj:
        raise VerificationError(f"(g) projection metadata does not re-derive: {proj} != {expected}")
    if proj["d"] != d:
        raise VerificationError(f"(g) projection d={proj['d']} != vector_dims d={d}")
    for field in ("source_pkl_sha256", "h5_manifest", "model", "tokenizer", "build_git_commit"):
        if not art.get(field):
            raise VerificationError(f"(g) metadata field {field!r} missing")
    wd = art["model"].get("weights_digest") if isinstance(art["model"], dict) else None
    if not (isinstance(wd, str) and len(wd) == 64 and all(c in "0123456789abcdef" for c in wd)):
        raise VerificationError(f"(g) model.weights_digest must be a full-content sha256, got {wd!r}")
    src_sha = libs.sha256_file(source_path)
    if art["source_pkl_sha256"] != src_sha:
        raise VerificationError("(g) source_pkl_sha256 does not match the given source pickle")

    # (a') (b) (c) (d) (e)
    shapes = {}
    n_with_ws = 0
    for e in entries:
        if e.checkpoint_id is not CheckpointID.CP2:
            raise VerificationError(f"(a') entry {e.id} carries checkpoint_id {e.checkpoint_id}")
        s = src_by_id[e.id]
        if not _payload_equal(e.payload, s.payload):
            raise VerificationError(f"(b) payload differs from source for {e.id}")
        for f in ("trajectory_id", "step_idx", "outcome"):
            if getattr(e, f, None) != getattr(s, f, None):
                raise VerificationError(f"(b) {f} differs from source for {e.id}")
        if list(e.prev_ids or []) != list(getattr(s, "prev_ids", []) or []) or \
                list(e.next_ids or []) != list(getattr(s, "next_ids", []) or []):
            raise VerificationError(f"(b) chain edges differ from source for {e.id}")
        for nb in list(e.prev_ids or []) + list(e.next_ids or []):
            if nb not in src_by_id:
                raise VerificationError(f"(c) edge {e.id} -> {nb} points outside the artifact")
        if set(e.query_keys) != {libs.FIELD}:
            raise VerificationError(f"(d) query_keys of {e.id} = {sorted(e.query_keys)}")
        k = _as_np(e.query_keys[libs.FIELD])
        if k.dtype != np.float32 or k.shape != (d,) or not np.all(np.isfinite(k)):
            raise VerificationError(f"(d) key of {e.id}: dtype={k.dtype} shape={k.shape} finite={np.all(np.isfinite(k))}")
        shape = tuple(_as_np(e.payload.action_chunk).shape)
        if shape != libs.ACTION_CHUNK_SHAPE:
            raise VerificationError(
                f"(e) action_chunk of {e.id} has shape {shape}, expected {libs.ACTION_CHUNK_SHAPE}"
            )
        shapes[shape] = shapes.get(shape, 0) + 1
        inter = e.payload.intermediates or {}
        if any(math.isclose(float(t), libs.WARM_START_T, abs_tol=1e-6) for t in inter):
            n_with_ws += 1
    if len(shapes) != 1:
        raise VerificationError(f"(e) heterogeneous action_chunk shapes {shapes}")
    if n_with_ws != len(entries):
        raise VerificationError(f"(e) only {n_with_ws}/{len(entries)} entries carry the start_t={libs.WARM_START_T} intermediate")

    # (a'') real backend round trip
    backend = InMemoryBackend(vector_dims={libs.FIELD: d})
    backend.load_artifact(cp2_path)
    storage = CacheStorage(backend)
    meta = storage.artifact_meta or {}
    if meta.get("projection") != proj or meta.get("id_policy") != libs.ID_POLICY:
        raise VerificationError("(a'') backend artifact_meta does not expose projection / id_policy")
    rng = random.Random(seed)
    sample = rng.sample(entries, min(search_samples, len(entries)))
    for e in sample:
        key = torch.as_tensor(_as_np(e.query_keys[libs.FIELD])).float()
        res = storage.search(cp2_query_spec(key))
        if not res:
            raise VerificationError(f"(a'') CP2 search returned nothing for {e.id}")
        if res[0].id != e.id and abs(float(res[0].score) - 1.0) > 1e-4:
            raise VerificationError(f"(a'') self-query of {e.id} returned {res[0].id} score={res[0].score}")
    return {
        "cp2_pkl": str(cp2_path), "source_pkl": str(source_path), "n_entries": len(entries),
        "d": d, "action_chunk_shape": next(iter(shapes)), "search_samples": len(sample),
        "projection_digest": proj["digest"], "ok": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cp2-pkl", required=True)
    ap.add_argument("--source-pkl", required=True)
    ap.add_argument("--search-samples", type=int, default=20)
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    rec = verify(args.cp2_pkl, args.source_pkl, search_samples=args.search_samples)
    if args.out:
        libs.dump_json(args.out, rec)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
