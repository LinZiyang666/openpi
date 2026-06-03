"""PrenormDotBackend — ROUND 2 prenorm-dot subclass of PrebuiltMatrixBackend.

ROUND 2 second cut of the cp1_search progressive tuning. Eliminates the dominant cost
located by Round-1's micro_bench: ``F.cosine_similarity`` is ~24x slower than a pure GEMV
on CPU (broadcast + per-call norm recompute + no BLAS; mv_only 1.2ms vs cosine_one 29ms on
the N=399 bucket). This subclass L2-row-normalizes each cosine-field bucket matrix ONCE at
prebuild, so cosine collapses to a single fp32 ``torch.mv`` (BLAS GEMV) at search;
robot_state (l2) keeps ``torch.norm`` verbatim (micro_bench shows l2 ~0.01-0.03ms, not a
bottleneck).

Inherits the Round-1 ``PrebuiltMatrixBackend`` machinery UNCHANGED: per-(checkpoint,
task_key, field) prebuilt matrices, the id->row map, the ``_fast_path_key`` safety window,
and the zero-copy (whole bucket in order) vs ``index_select`` (subset) gather. The ONLY
change vs Round 1 is the per-field operator (cosine: mv on the unit matrix; l2: torch.norm
on the raw matrix).

NOT bit-identical to the parent (Round 1 was). prenorm-dot vs ``F.cosine_similarity``
differ at the ~2e-6 raw level (fp32 rsqrt on near-unit vision vectors), amplified by the
zscore normalizer to ~1e-4 fused. fp32 is FORCED (reduced precision flips 3-way verdicts:
fp16=15/2640, bf16=194/2640 on record). Equivalence is therefore proven by 3-way verdict +
winner parity on a near-threshold subset, NOT by bit-identity.

A guarded top-K fp32 rescore lever (``rescore_top_k`` + ``keep_raw``) is reserved for the
case where the subset surfaces any near-boundary flip or bit-parity-on-winner is required;
it is DEFAULT OFF and currently raises if requested (deferred until a flip is observed —
the off path is expected to be 0-flip per the on-disk 0/2640 fp32 record). ``keep_raw``
already retains the pre-norm cosine matrices so the lever can be implemented without a
re-prebuild.

Frozen-library contract: assumes ``write_policy: never``. Matrices snapshot ``_entries``
once and the cosine buckets are normalized in place; any post-prebuild vector mutation
would serve stale rows.

Dependencies: ``exp.cache_latency_bench.opt.prebuilt_matrix_backend.PrebuiltMatrixBackend``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from exp.cache_latency_bench.opt.prebuilt_matrix_backend import PrebuiltMatrixBackend


class PrenormDotBackend(PrebuiltMatrixBackend):
    """PrebuiltMatrixBackend with L2-prenormalized cosine buckets scored by a single GEMV.

    ROUND 2 scope: only ``_prebuild_task_matrices`` and ``_compute_field_scores`` are
    overridden; the gather / zero-copy / safety-window machinery is inherited.
    """

    def __init__(
        self,
        vector_dims: dict[str, int],
        cosine_fields=frozenset({"vision_0", "vision_1"}),
        l2_fields=frozenset({"robot_state"}),
        rescore_top_k: int = 0,
        keep_raw: bool = False,
    ) -> None:
        super().__init__(vector_dims)
        if rescore_top_k > 0 and not keep_raw:
            raise ValueError("rescore_top_k>0 requires keep_raw=True (rescore needs raw cosine rows)")
        self._cosine_fields = frozenset(cosine_fields)
        self._l2_fields = frozenset(l2_fields)
        # A field cannot be both normalized (cosine) and scored raw (l2): a cosine field
        # typed l2 would read the unit matrix and break l2 bit-identity (G2 MINOR).
        assert self._cosine_fields.isdisjoint(self._l2_fields), "cosine_fields and l2_fields must be disjoint"
        self._rescore_top_k = int(rescore_top_k)
        self._keep_raw = bool(keep_raw)
        self._unit_keys: set = set()   # (ckpt, task, field) buckets that were L2-normalized
        self._raw_mat: dict = {}        # only if keep_raw: pre-norm cosine matrices (for rescore)
        self._prenorm_hits: int = 0     # diagnostics: prenorm GEMV path fired (cosine)

    # ------------------------------------------------------------------
    # Prebuild: normalize cosine buckets once
    # ------------------------------------------------------------------

    def _prebuild_task_matrices(self) -> None:
        """Build raw matrices (parent) then L2-row-normalize the cosine buckets in place.

        The normalization REPLACES the cosine bucket matrices (normed-only, +0 GB steady
        state); the l2 bucket (robot_state) is left raw. ``entry.query_keys`` are NOT
        touched (only the prebuilt ``self._mat`` copies are normalized), so the parent
        fallback path — which reads ``entry.query_keys`` — still computes raw cosine.
        """
        super()._prebuild_task_matrices()  # self._mat (raw) + self._id2row
        for key, mat in list(self._mat.items()):
            field = key[2]
            if field in self._cosine_fields:
                m = mat.float()
                if self._keep_raw:
                    self._raw_mat[key] = m.clone().contiguous()
                self._mat[key] = F.normalize(m, p=2, dim=1).contiguous()
                self._unit_keys.add(key)
        # Assert every cosine bucket got normalized and is fp32.
        for key, mat in self._mat.items():
            if key[2] in self._cosine_fields:
                assert key in self._unit_keys, f"cosine bucket {key} not normalized"
                assert mat.dtype == torch.float32, f"cosine bucket {key} not fp32"

    # ------------------------------------------------------------------
    # Override: cosine -> GEMV on unit matrix; l2 -> torch.norm; else parent
    # ------------------------------------------------------------------

    def _compute_field_scores(
        self,
        query_vec: torch.Tensor,
        candidates: list,
        field_name: str,
        sim_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one field. Whitelisted cosine -> fp32 GEMV on the unit bucket; l2 ->
        torch.norm on the raw bucket; anything else -> parent (which handles raw cosine /
        cross-bucket / memo fallback). fp32 forced.
        """
        if self._rescore_top_k > 0:
            raise NotImplementedError(
                "rescore_top_k lever is reserved but not implemented this round; the "
                "default off path is expected 0-flip. Enable only after a subset flip."
            )
        n = len(candidates)
        valid_indices = [i for i, e in enumerate(candidates) if field_name in e.query_keys]
        if not valid_indices:
            return torch.zeros(n), torch.zeros(n)

        sim_type = sim_cfg.get("type", "cosine")
        key = self._fast_path_key(candidates, valid_indices, field_name, sim_type)
        prenorm_ok = (
            key is not None
            and sim_type == "cosine"
            and field_name in self._cosine_fields
            and key in self._unit_keys
        )
        l2_ok = key is not None and sim_type == "l2" and field_name in self._l2_fields
        if not (prenorm_ok or l2_ok):
            # key None (unsafe window) OR cosine on a non-whitelisted (raw) bucket.
            return super()._compute_field_scores(query_vec, candidates, field_name, sim_cfg)

        rowmap = self._id2row[key]
        rows = [rowmap[candidates[i].id] for i in valid_indices]
        full = self._mat[key]
        if rows == list(range(full.shape[0])):
            mat = full                                              # [N, D] resident
        else:
            mat = full.index_select(0, torch.tensor(rows, dtype=torch.long)).contiguous()

        if prenorm_ok:
            qn = query_vec.float()
            qn = qn / qn.norm().clamp_min(1e-12)                    # unit query
            valid_scores = mat.mv(qn)                               # fp32 GEMV = cosine (unit rows)
            self._prenorm_hits += 1
        else:  # l2_ok — raw bucket, identical operator to the parent
            valid_scores = torch.norm(query_vec.float().unsqueeze(0) - mat, p=2, dim=1)

        scores = torch.zeros(n)
        mask = torch.zeros(n)
        idx_t = torch.tensor(valid_indices, dtype=torch.long)
        scores[idx_t] = valid_scores
        mask[idx_t] = 1.0
        return scores, mask
