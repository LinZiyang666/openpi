"""MlpRouterJudge — the X14 online-RL router baseline, living at the verdict layer.

Why this exists
---------------
TIER routes every control step with retrieval similarity + two thresholds. The
standing reviewer question is "why not just train a router?". The semantically
correct answer is that supervised per-step routing labels do not exist, so the
only sound training route is online RL — and this module is that baseline, run
for real: a 2-layer MLP occupies the judge slot, samples one arm per verdict,
and is trained by batch on-policy REINFORCE from the episode outcome.

Information contract (the point of the whole experiment)
--------------------------------------------------------
The MLP sees *exactly* what TIER's retrieval sees before it touches the
library: the post-``build()`` ``query_keys``. It is blind to every library-side
quantity — similarity scores, retrieved ids, payloads, neighbour history. That
blindness is structural, not conventional: ``_decide`` takes features only, and
the winner id is picked *after* the arm is chosen. Retrieval still runs (the
cache arm needs a payload to replay), it simply never reaches the network.

Arms and how a verdict maps onto the cache framework
-----------------------------------------------------
====== ========================================================================
arm    verdict
====== ========================================================================
teacher ``MISS`` — full Pi0.5 inference (the interceptor's normal miss path).
student ``FULL_HIT(winner_id=None, hit_override=True)`` — payloadless: the
        wired ``hit_executor`` sidecar produces the action, zero fetch.
cache   ``FULL_HIT(winner_id=results[0].id, hit_override=False)`` — forced
        replay of the cached clean action, even if a hit_executor is wired.
        Empty library degrades to ``MISS`` with ``fallback=true``.
====== ========================================================================

Dump layout (consumed by ``exp/rl_router`` trainers)
----------------------------------------------------
One directory per ``(run_id, batch_id)``; one shard + one sidecar per episode::

    <dump_dir>/<run_id>/<batch_id>/<stem>.bin      headerless raw fp16 [rows, dim]
    <dump_dir>/<run_id>/<batch_id>/<stem>.jsonl    per-step metadata (no features)
    <dump_dir>/<run_id>/<batch_id>/manifest.jsonl  append-only completion ledger
    <dump_dir>/_orphan/<uuid>.jsonl                identity-less rows (never trained on)
    <dump_dir>/_quarantine/                        swept ``.tmp`` from a crashed process

The manifest — not the conductor journal — is the authority on batch
completeness: a journal terminal record does not imply the shard was finalized.

Coupling map:
  DEPENDS ON: components/judge.py (JudgeResult / HitType contract),
              cache/types.py (query-key field names).
  CONSUMED BY: cache/config.py (`_build_judge` type "mlp_router"),
               cache/orchestrator.py (conditional ``query_keys`` injection +
               episode/task-end finalize broadcast),
               exp/rl_router/* (weights format, dump schema, encoder parity).
  IF CHANGED: the on-policy parity contract (§3.0) and the trainer's bitwise
              logprob verification break — bump ``encoder_version``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import threading
import time
import uuid
from typing import Optional

import torch

from openpi.cache.components.judge import HitType, JudgeResult
from openpi.cache.storage_types import RetrievalSignals, SearchResultLite
from openpi.cache.types import (
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

logger = logging.getLogger("openpi.cache.mlp_router")

# ----------------------------------------------------------------------------
# Frozen constants (any change here invalidates encoder_version / parity)
# ----------------------------------------------------------------------------

# Canonical concatenation order. A config may enable a subset; the encoder
# always emits them in this order so the feature vector layout is a property of
# the field *set*, never of how the yaml happened to list them.
CANONICAL_FIELD_ORDER: tuple[str, ...] = (VISION_0, VISION_1, VISION_2, ROBOT_STATE)

# Arm alphabet. The config's ``arms`` string selects one row.
ARM_SETS: dict[str, tuple[str, ...]] = {
    "ts": ("teacher", "student"),
    "tc": ("teacher", "cache"),
    "tsc": ("teacher", "student", "cache"),
}

# Dump feature dtype. The MLP decides on the *upcast* of these bytes, so the
# dump is a lossless record of the network's actual input (§3.0 parity).
FEATURE_DTYPE = torch.float16

# Every router forward runs single-threaded so the trainer's CPU reference can
# reproduce the logits bitwise. Intra-op thread count changes the reduction
# order of a matmul, which changes fp32 rounding — a tolerance-free ``==``
# verification cannot survive that.
ROUTER_TORCH_THREADS = 1

# Constant-arm logits: finite (JSON-serialisable) but far enough apart that
# softmax is exactly one-hot in fp32, so logprob(chosen) == 0.0.
_CONST_LOGIT_HI = 0.0
_CONST_LOGIT_LO = -1e9

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Manifest appends are the atomic persistence unit and are shared by every
# per-connection judge instance in this process.
_MANIFEST_LOCK = threading.Lock()

# One-shot guards, keyed so repeated per-connection construction is cheap.
_SWEPT_DIRS: set[str] = set()
_SWEEP_LOCK = threading.Lock()
_THREADS_PINNED = False


def pin_router_threads() -> None:
    """Pin torch intra-op threads so router forwards are bitwise reproducible.

    Called by the judge when a dump is configured (i.e. training arms, where the
    trainer verifies behaviour logprobs bitwise) and by ``exp/rl_router`` on the
    trainer side, so both ends share one definition of "the reference device".
    Idempotent and process-global by design — ``torch.set_num_threads`` is not
    scoped, and toggling it per call would let a concurrent connection observe
    the un-pinned state mid-forward.

    Evaluation arms (no dump) never call this, so a routed eval server keeps its
    normal CPU threading.
    """
    global _THREADS_PINNED
    if _THREADS_PINNED:
        return
    torch.set_num_threads(ROUTER_TORCH_THREADS)
    _THREADS_PINNED = True
    logger.info(
        "mlp_router: pinned torch intra-op threads to %d for bitwise logprob parity",
        ROUTER_TORCH_THREADS,
    )


# ----------------------------------------------------------------------------
# Feature encoder
# ----------------------------------------------------------------------------


class RouterFeatureEncoder:
    """Turn ``query_keys`` into the exact fp32 vector the MLP consumes.

    Operator order is frozen (§3.0)::

        raw fp32 -> concatenate -> affine normalize -> Q(fp32 -> fp16 -> fp32)

    and the MLP decides on the Q output, so dumping the fp16 tensor records the
    network's input losslessly and on-policy parity holds by construction.

    Two encoder generations share this class:

      - **v0-raw** (``mu``/``sigma`` are None): plain concatenation. Used by the
        warm-start collection pass, which runs before any statistics exist —
        this is what breaks the mu/sigma circular dependency.
      - **v2**: v0 followed by one affine over the WHOLE concatenated vector.

    The affine spans every field, not just robot_state. The original design
    normalised robot_state alone, on the premise that "the key builder already
    scales the vision fields"; measurement disproved it — the production
    ``cp1_spatial_pool_16`` keys have std ~5.25 and range +-209. At this width
    (65,568 inputs) that is fatal rather than untidy: Adam displaces every
    weight by ~lr per step regardless of gradient size, so one step moves a
    pre-activation by ~``lr * sum_j |x_j|``, which was ~82 for the frozen
    trainer lr and saturated the policy to a single arm in ONE update
    (student logit 0.84 -> -45.5, measured). Scaling the input is what lets the
    frozen §3.5 constants do sensible work: the same 40 updates then move the
    student rate 0.767 -> 0.637 smoothly instead of off a cliff.

    Because ``sigma`` carries an arbitrary positive scale, the ``1/sqrt(D)``
    that brings ``||x||`` to O(1) is folded into it by the fitter rather than
    living as a separate operator — one affine, one version hash, nothing to
    keep in sync.
    """

    def __init__(
        self,
        fields: tuple[str, ...],
        *,
        mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> None:
        if not fields:
            raise ValueError("RouterFeatureEncoder requires at least one field")
        unknown = [f for f in fields if f not in CANONICAL_FIELD_ORDER]
        if unknown:
            raise ValueError(
                f"unknown feature field(s) {unknown}; valid: {list(CANONICAL_FIELD_ORDER)}"
            )
        if len(set(fields)) != len(fields):
            raise ValueError(f"duplicate feature fields: {list(fields)}")
        if (mu is None) != (sigma is None):
            raise ValueError("normalization mu and sigma must be provided together")
        self._fields: tuple[str, ...] = tuple(
            f for f in CANONICAL_FIELD_ORDER if f in set(fields)
        )
        self._mu = None if mu is None else mu.detach().to(torch.float32).reshape(-1)
        self._sigma = None if sigma is None else sigma.detach().to(torch.float32).reshape(-1)
        if self._sigma is not None:
            if bool(torch.any(self._sigma <= 0)):
                raise ValueError("normalization sigma must be strictly positive")
            if self._mu.numel() != self._sigma.numel():
                raise ValueError(
                    f"mu dim {self._mu.numel()} != sigma dim {self._sigma.numel()}"
                )
        self._version = self._compute_version()

    # -- properties ---------------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Fields in canonical concatenation order."""
        return self._fields

    @property
    def version(self) -> str:
        """``encoder_version``: sha256 over field order + normalization params.

        Deliberately excludes the field *dimensions*: those are validated
        separately against the weights meta, and a dimension change is already
        caught there with a far clearer error.
        """
        return self._version

    @property
    def normalized(self) -> bool:
        """True for v2 (whole-vector affine active), False for v0-raw."""
        return self._mu is not None

    # -- encoding -----------------------------------------------------------

    def encode(self, query_keys: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the fp16 feature vector; upcast it for the MLP input.

        fp16 is returned (rather than the fp32 upcast) because it is what gets
        dumped; ``.to(torch.float32)`` on the result is exact, so the caller
        holding both costs nothing.
        """
        parts: list[torch.Tensor] = []
        for name in self._fields:
            t = query_keys.get(name)
            if t is None:
                raise ValueError(
                    f"mlp_router feature field {name!r} missing from query_keys "
                    f"(present: {sorted(query_keys)}); check keys.*.enabled in the yaml"
                )
            parts.append(t.detach().to(torch.float32).reshape(-1).cpu())
        x = torch.cat(parts)
        if self._mu is not None:
            if x.numel() != self._mu.numel():
                raise ValueError(
                    f"feature dim {x.numel()} != normalization stats dim "
                    f"{self._mu.numel()}; the weights were fitted on a different artifact"
                )
            x = (x - self._mu) / self._sigma
        return x.to(FEATURE_DTYPE)

    def field_dims(self, query_keys: dict[str, torch.Tensor]) -> dict[str, int]:
        """Per-field element counts for the current query, for meta validation."""
        return {
            name: int(query_keys[name].reshape(-1).numel())
            for name in self._fields
            if name in query_keys
        }

    # -- internals ----------------------------------------------------------

    def _compute_version(self) -> str:
        payload = {
            "fields": list(self._fields),
            "dtype": "float16",
            "op_order": "concat->normalize->quantize",
            "norm": "v0-raw" if self._mu is None else {
                "mu_sha": _tensor_sha(self._mu),
                "sigma_sha": _tensor_sha(self._sigma),
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    ).hexdigest()


# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------


class RouterWeights:
    """Validated 2-layer MLP parameters plus the meta they were saved with.

    Blob schema (``torch.save`` of a plain dict)::

        {"W1": [H, D], "b1": [H], "W2": [A, H], "b2": [A],
         "feature_mu": [D] | None, "feature_sigma": [D] | None,
         "meta": {"fields": [...], "dims": {...}, "arms": "tsc", "hidden": H,
                  "encoder_version": ..., "weights_version": ..., "model_sha": ...}}

    Loading is fail-fast on every meta field: a silently mismatched router is
    indistinguishable from a working one until the results are already wrong.
    """

    def __init__(self, blob: dict, *, path: str) -> None:
        self._path = path
        meta = blob.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"router weights {path!r}: missing 'meta' dict")
        for key in ("fields", "dims", "arms", "hidden", "encoder_version",
                    "weights_version", "model_sha"):
            if key not in meta:
                raise ValueError(f"router weights {path!r}: meta is missing {key!r}")
        self.meta = dict(meta)
        self.fields: tuple[str, ...] = tuple(meta["fields"])
        self.dims: dict[str, int] = {k: int(v) for k, v in dict(meta["dims"]).items()}
        self.arms: str = str(meta["arms"])
        self.hidden: int = int(meta["hidden"])
        self.encoder_version: str = str(meta["encoder_version"])
        self.weights_version: str = str(meta["weights_version"])

        try:
            self.W1 = blob["W1"].detach().to(torch.float32)
            self.b1 = blob["b1"].detach().to(torch.float32)
            self.W2 = blob["W2"].detach().to(torch.float32)
            self.b2 = blob["b2"].detach().to(torch.float32)
        except KeyError as exc:
            raise ValueError(f"router weights {path!r}: missing tensor {exc}") from exc

        # ``robot_state_*`` is the pre-v2 name, when the affine covered only
        # that field. Reading it here turns an old checkpoint into the explicit
        # dim-mismatch error below instead of a bare KeyError.
        mu = blob.get("feature_mu", blob.get("robot_state_mu"))
        sigma = blob.get("feature_sigma", blob.get("robot_state_sigma"))
        self.mu = None if mu is None else mu.detach().to(torch.float32)
        self.sigma = None if sigma is None else sigma.detach().to(torch.float32)

        expected_d = sum(self.dims[f] for f in self.fields)
        n_arms = len(ARM_SETS[self.arms]) if self.arms in ARM_SETS else -1
        if n_arms < 0:
            raise ValueError(
                f"router weights {path!r}: meta.arms {self.arms!r} not in {sorted(ARM_SETS)}"
            )
        for name, tensor, shape in (
            ("W1", self.W1, (self.hidden, expected_d)),
            ("b1", self.b1, (self.hidden,)),
            ("W2", self.W2, (n_arms, self.hidden)),
            ("b2", self.b2, (n_arms,)),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"router weights {path!r}: {name} has shape {tuple(tensor.shape)}, "
                    f"expected {shape}"
                )

        actual_sha = self.model_sha()
        if actual_sha != str(meta["model_sha"]):
            raise ValueError(
                f"router weights {path!r}: model_sha mismatch — meta says "
                f"{meta['model_sha']!r}, tensors hash to {actual_sha!r}"
            )

    def model_sha(self) -> str:
        """sha256 over the four parameter tensors in a fixed order."""
        return _params_sha(self.W1, self.b1, self.W2, self.b2)

    @classmethod
    def load(cls, path: str) -> "RouterWeights":
        """Load and fully validate a weights file (fail-fast at build time)."""
        # weights_only=True: the blob is plain tensors + primitive meta, so the
        # safe loader suffices and no artifact can execute code at load time.
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict):
            raise ValueError(
                f"router weights {path!r}: expected a dict, got {type(blob).__name__}"
            )
        return cls(blob, path=path)


def save_router_weights(
    path: str | os.PathLike,
    *,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    arms: str,
    fields: tuple[str, ...],
    dims: dict[str, int],
    weights_version: str,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
) -> dict:
    """Write a router checkpoint and return the meta that was stamped into it.

    The single writer for every producer — warm-start fitting, RL training, and
    tests. ``encoder_version`` and ``model_sha`` are derived here by the same
    code that ``RouterWeights`` validates against, so a producer cannot drift
    from the consumer's expectations; a mismatch would otherwise only surface as
    a fail-fast at server start, halfway into an unattended run.

    Writing is atomic so a crash mid-save cannot leave the serving side pointing
    at a truncated checkpoint.
    """
    encoder = RouterFeatureEncoder(fields, mu=mu, sigma=sigma)
    meta = {
        "fields": list(encoder.fields),
        "dims": {k: int(v) for k, v in dims.items()},
        "arms": str(arms),
        "hidden": int(b1.numel()),
        "encoder_version": encoder.version,
        "weights_version": str(weights_version),
        "model_sha": _params_sha(W1, b1, W2, b2),
    }
    blob = {
        "W1": W1.detach().to(torch.float32),
        "b1": b1.detach().to(torch.float32),
        "W2": W2.detach().to(torch.float32),
        "b2": b2.detach().to(torch.float32),
        "feature_mu": None if mu is None else mu.detach().to(torch.float32),
        "feature_sigma": None if sigma is None else sigma.detach().to(torch.float32),
        "meta": meta,
    }
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, target)
    return meta


def _params_sha(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Per-episode dump buffer
# ----------------------------------------------------------------------------


class _EpisodeBuffer:
    """In-memory accumulation for one episode, flushed once at finalize.

    Buffering (rather than streaming) is what makes the shard atomic: a stale
    attempt that dies mid-episode never leaves a half-written ``.bin`` that a
    later reader could mistake for training data. At ~131 KB/step and <= ~200
    steps this is ~26 MB per in-flight episode.
    """

    __slots__ = ("identity", "seed_ep", "rows", "features")

    def __init__(self, identity: Optional[dict], seed_ep: Optional[int]) -> None:
        self.identity = identity
        self.seed_ep = seed_ep
        self.rows: list[dict] = []
        self.features: list[torch.Tensor] = []


# ----------------------------------------------------------------------------
# The judge
# ----------------------------------------------------------------------------


class MlpRouterJudge:
    """Verdict-layer MLP router (config ``judge.type: "mlp_router"``).

    Two mutually exclusive modes, enforced by ``validate_cache_config``:

      - ``weights_path``: a trained/warm-started network decides.
      - ``constant_arm``: every verdict picks one fixed arm with constant
        logits. Used by the warm-start collection pass, which needs on-policy
        features and outcomes but no network (and hence no normalization
        statistics — that is what breaks the mu/sigma circular dependency).

    Sampling is per-episode reproducible: ``on_episode_start`` derives
    ``seed_ep = sha256(run_seed, task_uid, attempt, weights_version)`` and seeds
    a private ``torch.Generator``. Replaying the same identity replays the same
    arm sequence, so an interrupted run and its resume are equivalent without
    persisting any live RNG state. An episode whose identity is incomplete is
    marked ``identity=missing``: it is forced to argmax and its rows are
    isolated, never trained on.
    """

    def __init__(
        self,
        *,
        arms: str = "tsc",
        weights_path: Optional[str] = None,
        constant_arm: Optional[str] = None,
        feature_fields: Optional[list[str]] = None,
        hidden: int = 256,
        temperature: float = 1.0,
        mode: str = "sample",
        dump_dir: str = "",
        seed: int = 0,
    ) -> None:
        if arms not in ARM_SETS:
            raise ValueError(f"arms must be one of {sorted(ARM_SETS)}, got {arms!r}")
        if (weights_path is None) == (constant_arm is None):
            raise ValueError(
                "exactly one of weights_path / constant_arm must be set "
                f"(got weights_path={weights_path!r}, constant_arm={constant_arm!r})"
            )
        if mode not in ("sample", "argmax"):
            raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")

        self._arms: tuple[str, ...] = ARM_SETS[arms]
        self._arms_key = arms
        self._mode = mode
        self._temperature = float(temperature)
        self._run_seed = int(seed)
        self._dump_dir = str(dump_dir or "")
        self._hidden = int(hidden)

        fields = tuple(feature_fields) if feature_fields else (VISION_0, VISION_1, ROBOT_STATE)

        if constant_arm is not None:
            if constant_arm not in self._arms:
                raise ValueError(
                    f"constant_arm {constant_arm!r} is not in arms {self._arms}"
                )
            if mode != "argmax":
                raise ValueError("constant_arm requires mode='argmax'")
            self._weights: Optional[RouterWeights] = None
            self._encoder = RouterFeatureEncoder(fields)
            self._weights_version = f"constant-{constant_arm}"
            self._const_arm_idx = self._arms.index(constant_arm)
        else:
            weights = RouterWeights.load(str(weights_path))
            if weights.arms != arms:
                raise ValueError(
                    f"router weights arms {weights.arms!r} != configured arms {arms!r}"
                )
            if weights.hidden != self._hidden:
                raise ValueError(
                    f"router weights hidden {weights.hidden} != configured hidden {self._hidden}"
                )
            self._weights = weights
            self._encoder = RouterFeatureEncoder(fields, mu=weights.mu, sigma=weights.sigma)
            if tuple(weights.fields) != self._encoder.fields:
                raise ValueError(
                    f"router weights fields {tuple(weights.fields)} != configured "
                    f"fields {self._encoder.fields} (canonical order)"
                )
            if self._encoder.version != weights.encoder_version:
                raise ValueError(
                    f"router weights encoder_version {weights.encoder_version!r} != "
                    f"encoder built from this config {self._encoder.version!r}"
                )
            self._weights_version = weights.weights_version
            self._const_arm_idx = -1

        # Per-episode state.
        self._buffer: Optional[_EpisodeBuffer] = None
        self._generator: Optional[torch.Generator] = None
        self._decision_idx = 0
        self._dims_checked = False

        if self._dump_dir:
            # Parity matters only where a trainer verifies logprobs, i.e. where
            # a dump exists. Eval arms keep the process's normal threading.
            pin_router_threads()
            _sweep_stale_tmp(self._dump_dir)

    # ------------------------------------------------------------------
    # Introspection (used by config golden tests and the emitters)
    # ------------------------------------------------------------------

    @property
    def arms(self) -> tuple[str, ...]:
        """Ordered arm names for this variant."""
        return self._arms

    @property
    def weights_version(self) -> str:
        """Authoritative weights version stamped onto every verdict and shard."""
        return self._weights_version

    @property
    def encoder_version(self) -> str:
        """Authoritative encoder version stamped onto every dump row."""
        return self._encoder.version

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self, *, extra_metadata: Optional[dict] = None) -> None:
        """Seed this episode's RNG and open a fresh dump buffer.

        A buffer left open by the previous episode means its ``episode_end``
        never arrived (the client's ``episode_end`` RPC is best-effort — see
        ``episode_runner.py``'s ``contextlib.suppress``). Close it as
        ``partial`` before opening the new one so the batch completeness check
        sees the truth instead of silently inheriting rows.
        """
        self._finalize("partial")
        identity = self._extract_identity(dict(extra_metadata or {}))
        seed_ep = None if identity is None else self._derive_seed(identity)
        if seed_ep is None:
            self._generator = None
        else:
            gen = torch.Generator()
            gen.manual_seed(seed_ep)
            self._generator = gen
        self._decision_idx = 0
        self._buffer = _EpisodeBuffer(identity, seed_ep)

    def on_episode_end(self) -> None:
        """Finalize the episode's shard. Broadcast from Orchestrator's ``finally``."""
        self._finalize("complete")

    def on_task_end(self) -> None:
        """Connection teardown: whatever is still open ended without an episode_end."""
        self._finalize("partial")

    # ------------------------------------------------------------------
    # Verdict path
    # ------------------------------------------------------------------

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view=None,
        history=None,
        retrieval_signals: Optional[RetrievalSignals] = None,
        query_keys: Optional[dict[str, torch.Tensor]] = None,
    ) -> JudgeResult:
        """Sample one arm and map it onto a cache verdict.

        ``results`` is touched for exactly two purposes, both *after* the arm is
        chosen: picking the cache arm's winner id, and detecting an empty
        library. ``view`` / ``history`` / ``retrieval_signals`` / ``cached_data``
        are accepted (the Orchestrator injects them unconditionally) and never
        read — that is the masking contract, and it is locked by a test that
        permutes the scores and asserts the decision is unchanged.
        """
        del cached_data, view, history, retrieval_signals  # masked by contract
        if query_keys is None:
            raise ValueError(
                "mlp_router requires query_keys; Orchestrator injects them when the "
                "judge signature declares the parameter (see judge_accepts_query_keys)"
            )

        features_16 = self._maybe_encode(query_keys)
        features_32 = None if features_16 is None else features_16.to(torch.float32)
        arm_idx, logits = self._decide(features_32)

        probs = torch.softmax(logits / self._temperature, dim=0)
        logprob = float(
            torch.log_softmax(logits / self._temperature, dim=0)[arm_idx].item()
        )
        arm_sampled = self._arms[arm_idx]

        # ---- arm -> verdict -------------------------------------------------
        fallback = False
        if arm_sampled == "student":
            hit_type, winner_id, override = HitType.FULL_HIT, None, True
            arm_mapped = "student"
        elif arm_sampled == "cache":
            if results:
                hit_type, winner_id, override = HitType.FULL_HIT, results[0].id, False
                arm_mapped = "cache"
            else:
                # Cold / empty library: there is nothing to replay. Degrade to
                # teacher and record it, so cost accounting bills the arm that
                # actually ran and the fallback rate is auditable (R12).
                hit_type, winner_id, override = HitType.MISS, None, None
                arm_mapped, fallback = "teacher", True
        else:
            hit_type, winner_id, override = HitType.MISS, None, None
            arm_mapped = "teacher"

        decision_idx = self._decision_idx
        self._decision_idx += 1

        router_outputs = {
            "decision_idx": decision_idx,
            "arm_sampled": arm_sampled,
            "arm_executed": None,  # stamped by the Interceptor on the executed path
            "probs": [float(p) for p in probs.tolist()],
            "temperature": self._temperature,
            "weights_version": self._weights_version,
            "seed_ep": None if self._buffer is None else self._buffer.seed_ep,
            "fallback": fallback,
        }
        self._record(decision_idx, logits, arm_sampled, arm_mapped, logprob, features_16)
        return JudgeResult(
            hit_type,
            winner_id,
            hit_override=override,
            router_outputs=router_outputs,
        )

    # ------------------------------------------------------------------
    # Decision (library-blind by signature)
    # ------------------------------------------------------------------

    def _decide(self, features: Optional[torch.Tensor]) -> tuple[int, torch.Tensor]:
        """Pick an arm from features alone. Returns ``(arm_idx, logits)``.

        The signature is the masking contract in executable form: no results, no
        view, no history, no retrieval signals, no cached data. A test asserts
        this statically so a future edit cannot quietly widen it.
        """
        logits = self._logits(features)
        if self._effective_mode() == "argmax":
            return int(torch.argmax(logits).item()), logits
        probs = torch.softmax(logits / self._temperature, dim=0)
        idx = int(torch.multinomial(probs, 1, generator=self._generator).item())
        return idx, logits

    def _logits(self, features: Optional[torch.Tensor]) -> torch.Tensor:
        if self._weights is None:
            logits = torch.full((len(self._arms),), _CONST_LOGIT_LO, dtype=torch.float32)
            logits[self._const_arm_idx] = _CONST_LOGIT_HI
            return logits
        if features is None:  # pragma: no cover - guarded by _maybe_encode
            raise RuntimeError("mlp_router: weights configured but no features encoded")
        w = self._weights
        # addmv(bias, mat, vec): [H] + [H, D] @ [D]. A matrix-vector product
        # parallelises over output rows, so each element's reduction order is
        # thread-count independent; pin_router_threads() removes the remaining
        # doubt for the bitwise parity contract.
        hidden = torch.addmv(w.b1, w.W1, features).clamp_min(0.0)
        return torch.addmv(w.b2, w.W2, hidden)

    def _effective_mode(self) -> str:
        """Sampling needs a seeded generator; without identity there is none."""
        if self._mode == "sample" and self._generator is not None:
            return "sample"
        return "argmax"

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def _maybe_encode(self, query_keys: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Encode when the decision or the dump needs it; otherwise skip entirely.

        A constant-arm eval arm with no dump touches no tensor at all — the
        router's own cost stays out of the measurement it is not part of.
        """
        if self._weights is None and not self._dump_dir:
            return None
        # Validate the observation space before touching it: a width mismatch
        # reported as "these dims are not the trained dims" beats the same fault
        # surfacing from inside the normalization step.
        if self._weights is not None and not self._dims_checked:
            self._check_dims(query_keys)
            self._dims_checked = True
        return self._encoder.encode(query_keys)

    def _check_dims(self, query_keys: dict[str, torch.Tensor]) -> None:
        actual = self._encoder.field_dims(query_keys)
        expected = self._weights.dims
        if actual != {k: int(v) for k, v in expected.items()}:
            raise ValueError(
                f"mlp_router feature dims {actual} do not match weights meta {expected}; "
                "the artifact this router was trained on has a different key builder"
            )

    # ------------------------------------------------------------------
    # Identity and RNG
    # ------------------------------------------------------------------

    def _extract_identity(self, extra: dict) -> Optional[dict]:
        """Return the five-part identity, or None when it is unusable.

        None means "this episode produces no training data": the judge falls
        back to argmax and its rows are isolated. Losing an episode is a
        recoverable batch gap (the packager schedules a repair); mixing an
        unidentifiable episode into a training batch is not recoverable.
        """
        missing = [k for k in ("run_id", "batch_id", "task_uid", "attempt") if extra.get(k) is None]
        if missing:
            if self._dump_dir:
                logger.error(
                    "mlp_router: episode identity incomplete (missing %s); forcing argmax "
                    "and isolating rows — this episode yields no training data", missing,
                )
            return None
        wire_version = extra.get("weights_version")
        if wire_version is not None and str(wire_version) != self._weights_version:
            # The task was dispatched against a different weights version than
            # this connection's bundle carries: a hot-swap race (R6). Do not
            # raise — a raise would fail an episode that the batch can still
            # repair; drop it from training instead and let the manifest gap
            # surface it.
            logger.error(
                "mlp_router: task weights_version %r != loaded %r; isolating episode",
                wire_version, self._weights_version,
            )
            return None
        try:
            attempt = int(extra["attempt"])
        except (TypeError, ValueError):
            logger.error("mlp_router: attempt %r is not an int; isolating episode",
                         extra.get("attempt"))
            return None
        return {
            "run_id": str(extra["run_id"]),
            "batch_id": str(extra["batch_id"]),
            "task_uid": str(extra["task_uid"]),
            "attempt": attempt,
            "weights_version": self._weights_version,
        }

    def _derive_seed(self, identity: dict) -> int:
        digest = hashlib.sha256(
            "|".join([
                str(self._run_seed),
                identity["task_uid"],
                str(identity["attempt"]),
                identity["weights_version"],
            ]).encode("utf-8")
        ).digest()
        # torch.Generator.manual_seed accepts a signed 64-bit value; mask to 63
        # bits so it is always positive.
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    # ------------------------------------------------------------------
    # Dump: buffering and finalize
    # ------------------------------------------------------------------

    def _record(
        self,
        decision_idx: int,
        logits: torch.Tensor,
        arm_sampled: str,
        arm_mapped: str,
        logprob: float,
        features: Optional[torch.Tensor],
    ) -> None:
        if not self._dump_dir or self._buffer is None:
            return
        ident = self._buffer.identity or {}
        self._buffer.rows.append({
            "run_id": ident.get("run_id"),
            "batch_id": ident.get("batch_id"),
            "task_uid": ident.get("task_uid"),
            "attempt": ident.get("attempt"),
            "weights_version": self._weights_version,
            "encoder_version": self._encoder.version,
            "seed_ep": self._buffer.seed_ep,
            "decision_idx": decision_idx,
            "logits": [float(x) for x in logits.tolist()],
            "temperature": self._temperature,
            "arm_sampled": arm_sampled,
            "arm_mapped": arm_mapped,
            "logprob_sampled": logprob,
            "ts": time.time(),
        })
        # Identity-less episodes keep their metadata (for auditing the loss) but
        # never their features: nothing downstream may train on them.
        if features is not None and self._buffer.identity is not None:
            self._buffer.features.append(features)

    def _finalize(self, status: str) -> None:
        """Flush the open buffer exactly once. Idempotent and never raises.

        A finalize failure must not propagate: it would abort the episode-end
        path for reasons unrelated to the episode's outcome. The cost of
        swallowing is a missing manifest entry, which is exactly the signal the
        batch completeness check is built to catch and repair.
        """
        buf, self._buffer = self._buffer, None
        self._generator = None
        if buf is None or not self._dump_dir:
            return
        if not buf.rows:
            # No verdict ran. Only worth a ledger entry if we know which slot it
            # was; otherwise there is nothing to say.
            if buf.identity is not None:
                try:
                    self._append_manifest(self._empty_manifest_entry(buf))
                except Exception:  # noqa: BLE001 - see docstring
                    logger.exception("mlp_router: failed to record empty episode")
            return
        try:
            if buf.identity is None:
                self._write_orphan(buf)
            else:
                self._write_shard(buf, status)
        except Exception:  # noqa: BLE001 - see docstring
            logger.exception(
                "mlp_router: failed to finalize episode dump (status=%s, identity=%s)",
                status, None if buf.identity is None else buf.identity.get("task_uid"),
            )

    def _write_shard(self, buf: _EpisodeBuffer, status: str) -> None:
        ident = buf.identity
        directory = pathlib.Path(self._dump_dir) / ident["run_id"] / ident["batch_id"]
        directory.mkdir(parents=True, exist_ok=True)
        stem = _shard_stem(ident)

        matrix = torch.stack(buf.features).contiguous()
        payload = matrix.numpy().tobytes()
        rows, dim = int(matrix.shape[0]), int(matrix.shape[1])
        sidecar_bytes = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in buf.rows
        ).encode("utf-8")

        # Sidecar first, shard second, manifest last: the manifest entry is the
        # only thing a reader trusts, so it must be the last byte written.
        _atomic_write(directory / f"{stem}.jsonl", sidecar_bytes)
        _atomic_write(directory / f"{stem}.bin", payload)

        self._append_manifest({
            **ident,
            "encoder_version": self._encoder.version,
            "seed_ep": buf.seed_ep,
            "shard": f"{stem}.bin",
            "sidecar": f"{stem}.jsonl",
            "rows": rows,
            "dim": dim,
            "dtype": "float16",
            "sha256": hashlib.sha256(payload).hexdigest(),
            # The sidecar carries the behaviour authority (logits / logprobs /
            # per-step identity), so it needs the same tamper gate as the
            # features — a trainer that trusts one and not the other can be fed
            # a mismatched pair.
            "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "status": status,
            "ts": time.time(),
        })

    def _empty_manifest_entry(self, buf: _EpisodeBuffer) -> dict:
        return {
            **buf.identity,
            "encoder_version": self._encoder.version,
            "seed_ep": buf.seed_ep,
            "shard": None,
            "sidecar": None,
            "rows": 0,
            "dim": 0,
            "dtype": "float16",
            "sha256": None,
            "sidecar_sha256": None,
            "status": "empty",
            "ts": time.time(),
        }

    def _append_manifest(self, entry: dict) -> None:
        ident_dir = (
            pathlib.Path(self._dump_dir) / str(entry["run_id"]) / str(entry["batch_id"])
        )
        ident_dir.mkdir(parents=True, exist_ok=True)
        path = ident_dir / "manifest.jsonl"
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _MANIFEST_LOCK, path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def _write_orphan(self, buf: _EpisodeBuffer) -> None:
        directory = pathlib.Path(self._dump_dir) / "_orphan"
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            directory / f"{uuid.uuid4().hex}.jsonl",
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in buf.rows).encode("utf-8"),
        )


# ----------------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------------


def _shard_stem(identity: dict) -> str:
    """Filesystem-safe, collision-free stem for one ``(uid, attempt, version)``.

    A stale attempt and the current one must never resolve to the same path —
    that is what keeps a late writer from overwriting a finalized shard. The
    sanitised name is for humans; the appended digest is what guarantees
    distinctness after sanitisation, and the manifest carries the authoritative
    identity fields regardless.
    """
    raw = f"{identity['task_uid']}__a{identity['attempt']}__{identity['weights_version']}"
    safe = _SAFE_NAME_RE.sub("_", raw)[:120]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{digest}"


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    """Write ``payload`` durably: tmp -> fsync -> atomic rename."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _sweep_stale_tmp(dump_dir: str) -> None:
    """Quarantine ``.tmp`` leftovers from a previous process, once per dir.

    A ``.tmp`` file can only outlive its writer if the process died inside
    ``_atomic_write``; it is by definition a torn write. Sweeping once per
    process (at the first judge construction) keeps it clear of any live
    finalize in this process, since no ``.tmp`` of ours exists yet.
    """
    with _SWEEP_LOCK:
        if dump_dir in _SWEPT_DIRS:
            return
        _SWEPT_DIRS.add(dump_dir)
    root = pathlib.Path(dump_dir)
    if not root.is_dir():
        return
    quarantine = root / "_quarantine"
    try:
        stale = [p for p in root.rglob("*.tmp") if quarantine not in p.parents]
        if not stale:
            return
        quarantine.mkdir(parents=True, exist_ok=True)
        for path in stale:
            target = quarantine / str(path.relative_to(root)).replace(os.sep, "__")
            counter = 0
            while target.exists():
                counter += 1
                target = target.with_name(f"{target.name}-{counter}")
            os.replace(path, target)
            logger.warning("mlp_router: quarantined torn write %s -> %s", path, target)
    except Exception:  # noqa: BLE001 - startup hygiene must not block serving
        logger.exception("mlp_router: tmp sweep failed under %s", dump_dir)


__all__ = [
    "ARM_SETS",
    "CANONICAL_FIELD_ORDER",
    "FEATURE_DTYPE",
    "ROUTER_TORCH_THREADS",
    "MlpRouterJudge",
    "RouterFeatureEncoder",
    "RouterWeights",
    "pin_router_threads",
    "save_router_weights",
]
