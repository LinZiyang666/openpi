"""Shared engine for the fusion-normalization theory study (offline, read-only).

Loads cache library artifacts (.pkl), computes exact query-vs-library raw
similarity matrices for the three production fields (vision_0/vision_1 cosine,
robot_state L2), and provides numpy ports of every Layer-1 normalizer family,
calibration fitters (legacy random-pair vs LOEO), squash variants, fusion, and
retrieval metrics with trajectory-cluster bootstrap CIs.

Public interface
----------------
- ``load_artifact`` / ``get_scores`` (cached matrix computation)
- ``fit_*`` calibration fitters and ``NORMALIZERS`` apply functions
- ``SQUASHES`` for the z-score squash ablation
- ``fuse_and_rank`` + ``retrieval_metrics`` + ``cluster_bootstrap``

Consumed by expA/expB/expC/expD scripts in this directory. Depends on torch,
numpy and (for parity checks only) ``openpi.cache.components.score_normalizers``.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = REPO / "exp/common/data/cache_artifacts"

FIELDS = ("vision_0", "vision_1", "robot_state")
# prompt_emb is excluded from the current fusion (plan D7) but kept here to
# reconstruct the legacy 4-field failure mode (task-constant -> denom<=0).
EXTRA_FIELDS = ("prompt_emb",)
SIM_TYPE = {"vision_0": "cosine", "vision_1": "cosine", "robot_state": "l2", "prompt_emb": "cosine"}
LEGACY_TAU = 0.334717  # exp/common/calibrate_score_sum_stats.py:44

# Production-optimum Layer-2 weights (RESULTS.md: spatial_16 v0@0.06 v1@0.50 rs@0.44)
W_PROD = {"vision_0": 0.06, "vision_1": 0.50, "robot_state": 0.44}
W_UNIF = {f: 1.0 / 3.0 for f in FIELDS}

COMBOS = [
    ("libero_spatial", "cp1_spatial_pool_16"),
    ("libero_spatial", "cp1_mean_pool"),
    ("libero_10", "cp1_spatial_pool_16"),
    ("libero_10", "cp1_mean_pool"),
]


# ------------------------------------------------------------------
# Artifact loading + exact score matrices (cached)
# ------------------------------------------------------------------
@dataclass
class Artifact:
    suite: str
    builder: str
    n: int
    task: np.ndarray        # [N] int task codes
    task_names: list[str]
    traj: np.ndarray        # [N] int trajectory codes
    step_idx: np.ndarray    # [N] int
    traj_len: np.ndarray    # [N] int (length of own trajectory)
    actions: np.ndarray     # [N, H*A] float32 flattened action chunks
    raw: dict[str, np.ndarray] = field(default_factory=dict)  # field -> [N,N]
    action_d2: np.ndarray | None = None  # [N,N] mean squared action distance

    @property
    def same_traj(self) -> np.ndarray:
        return self.traj[:, None] == self.traj[None, :]

    @property
    def same_task(self) -> np.ndarray:
        return self.task[:, None] == self.task[None, :]


def _pkl_path(suite: str, builder: str) -> Path:
    return ARTIFACT_ROOT / suite / f"{builder}.pkl"


def _cache_key(suite: str, builder: str) -> str:
    p = _pkl_path(suite, builder)
    st = p.stat()
    h = hashlib.md5(f"{p}:{st.st_size}:{st.st_mtime_ns}:v2".encode()).hexdigest()[:10]
    return f"{suite}__{builder}__{h}"


def load_artifact(suite: str, builder: str, cache_dir: Path, device: str | None = None) -> Artifact:
    """Load an artifact and its exact [N,N] raw score matrices (memoized on disk)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{_cache_key(suite, builder)}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        art = Artifact(
            suite=suite, builder=builder, n=int(z["n"]),
            task=z["task"], task_names=list(z["task_names"]),
            traj=z["traj"], step_idx=z["step_idx"], traj_len=z["traj_len"],
            actions=z["actions"],
        )
        art.raw = {f: z[f"raw_{f}"] for f in FIELDS + EXTRA_FIELDS}
        art.action_d2 = z["action_d2"]
        return art

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(_pkl_path(suite, builder), "rb") as fh:
        data = pickle.load(fh)
    entries = data["entries"]
    n = len(entries)

    task_names = sorted({e.payload.task_key for e in entries})
    tmap = {t: i for i, t in enumerate(task_names)}
    trajs = sorted({e.trajectory_id for e in entries})
    jmap = {t: i for i, t in enumerate(trajs)}
    task = np.array([tmap[e.payload.task_key] for e in entries], dtype=np.int32)
    traj = np.array([jmap[e.trajectory_id] for e in entries], dtype=np.int32)
    step_idx = np.array([e.step_idx for e in entries], dtype=np.int32)
    tl = {j: 0 for j in range(len(trajs))}
    for e in entries:
        j = jmap[e.trajectory_id]
        tl[j] = max(tl[j], e.step_idx + 1)
    traj_len = np.array([tl[j] for j in traj], dtype=np.int32)
    actions = np.stack(
        [np.asarray(e.payload.action_chunk, dtype=np.float32).reshape(-1) for e in entries]
    )

    art = Artifact(
        suite=suite, builder=builder, n=n, task=task, task_names=task_names,
        traj=traj, step_idx=step_idx, traj_len=traj_len, actions=actions,
    )
    with torch.no_grad():
        for f in FIELDS + EXTRA_FIELDS:
            vec = torch.stack(
                [torch.as_tensor(np.asarray(e.query_keys[f]), dtype=torch.float32) for e in entries]
            ).to(dev)
            if SIM_TYPE[f] == "cosine":
                vn = torch.nn.functional.normalize(vec, dim=1)
                m = vn @ vn.T
            else:
                m = torch.cdist(vec, vec, p=2)
            art.raw[f] = m.float().cpu().numpy()
            del vec, m
        a = torch.as_tensor(actions, device=dev)
        art.action_d2 = (torch.cdist(a, a, p=2) ** 2 / a.shape[1]).cpu().numpy()

    np.savez_compressed(
        cpath, n=n, task=task, task_names=np.array(task_names, dtype=object),
        traj=traj, step_idx=step_idx, traj_len=traj_len, actions=actions,
        action_d2=art.action_d2, **{f"raw_{f}": art.raw[f] for f in FIELDS + EXTRA_FIELDS},
    )
    return art


# ------------------------------------------------------------------
# Score pools (which pair distribution a calibration sees)
# ------------------------------------------------------------------
def pool_random_pair(raw: np.ndarray) -> np.ndarray:
    """All ordered i!=j pairs — deterministic version of the legacy random-pair
    sampling in calibrate_score_sum_stats.py (which includes intra-trajectory
    pairs and thus self-episode near-duplicates)."""
    n = raw.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return raw[mask]


def pool_loeo(raw: np.ndarray, same_traj: np.ndarray) -> np.ndarray:
    """Query -> library pairs with the query's own trajectory removed (serving
    distribution; mirrors collect_loeo_scores in calibrate_score_normalizers.py)."""
    return raw[~same_traj]


# ------------------------------------------------------------------
# Legacy direction-unify mapping (old pipeline step before percentiles)
# ------------------------------------------------------------------
def map_legacy(raw: np.ndarray, sim_type: str, tau: float = LEGACY_TAU) -> np.ndarray:
    if sim_type == "cosine":
        return (raw + 1.0) / 2.0
    return np.exp(-raw / tau)


def orient(raw: np.ndarray, sim_type: str) -> np.ndarray:
    return raw if sim_type == "cosine" else -raw


# ------------------------------------------------------------------
# Calibration fitters (numpy ports; formulas identical to src / legacy script)
# ------------------------------------------------------------------
def fit_legacy_percentile(pool: np.ndarray, sim_type: str, tau: float = LEGACY_TAU) -> dict:
    s0 = map_legacy(pool, sim_type, tau)
    return {"p5": float(np.percentile(s0, 5)), "p95": float(np.percentile(s0, 95)), "tau": tau}


def fit_zscore(pool: np.ndarray, sim_type: str) -> dict:
    x = orient(pool, sim_type)
    return {"mu": float(np.mean(x)), "sigma": float(np.std(x))}


def fit_affine_clip(pool: np.ndarray, sim_type: str, pct=(1.0, 99.0)) -> dict:
    x = orient(pool, sim_type)
    return {"lo": float(np.percentile(x, pct[0])), "hi": float(np.percentile(x, pct[1]))}


def fit_logit(pool: np.ndarray, sim_type: str, pct=(1.0, 99.0)) -> dict:
    assert sim_type == "cosine"
    return {"lo": float(np.percentile(pool, pct[0])), "hi": float(np.percentile(pool, pct[1])), "eps": 1e-4}


def fit_neg_log_one_minus(pool: np.ndarray, sim_type: str, pct=(1.0, 99.0)) -> dict:
    assert sim_type == "cosine"
    eps = 1e-4
    v = -np.log(np.clip(1.0 - pool + eps, eps, None))
    return {"v_lo": float(np.percentile(v, pct[0])), "v_hi": float(np.percentile(v, pct[1])), "eps": eps}


def fit_power(pool: np.ndarray, sim_type: str, pct=(1.0, 99.0)) -> dict:
    assert sim_type == "cosine"
    lo = float(np.percentile(pool, pct[0]))
    hi = float(np.percentile(pool, pct[1]))
    denom = hi - lo
    if denom <= 0:
        return {"lo": lo, "hi": hi, "gamma": 1.0}
    u_med = float(np.clip((np.median(pool) - lo) / denom, 1e-3, 1 - 1e-3))
    gamma = float(np.clip(math.log(0.5) / math.log(u_med), 0.1, 10.0))
    return {"lo": lo, "hi": hi, "gamma": gamma}


def fit_exp_l2(pool: np.ndarray, sim_type: str) -> dict:
    assert sim_type == "l2"
    med = float(np.median(pool))
    return {"tau": med / math.log(2.0) if med > 0 else 1.0}


# ------------------------------------------------------------------
# Normalizer apply functions (raw -> [0,1]); mirror src semantics exactly
# ------------------------------------------------------------------
def apply_legacy_percentile(raw, sim_type, p):
    s0 = map_legacy(raw, sim_type, p.get("tau", LEGACY_TAU))
    denom = p["p95"] - p["p5"]
    if denom <= 0:
        return np.full_like(s0, 0.5)
    return np.clip((s0 - p["p5"]) / denom, 0.0, 1.0)


def apply_zscore_tanh(raw, sim_type, p):
    x = orient(raw, sim_type)
    sigma = p["sigma"] if p["sigma"] > 1e-12 else 1.0
    return 0.5 * (np.tanh((x - p["mu"]) / sigma) + 1.0)


def apply_affine_clip(raw, sim_type, p):
    x = orient(raw, sim_type)
    denom = p["hi"] - p["lo"]
    if denom <= 0:
        return np.full_like(x, 0.5)
    return np.clip((x - p["lo"]) / denom, 0.0, 1.0)


def apply_logit(raw, sim_type, p):
    denom = p["hi"] - p["lo"]
    if denom <= 0:
        return np.full_like(raw, 0.5)
    eps = p["eps"]
    u = np.clip((raw - p["lo"]) / denom, eps, 1 - eps)
    t = np.log(u / (1 - u))
    t_lo = math.log(eps / (1 - eps))
    t_hi = math.log((1 - eps) / eps)
    return np.clip((t - t_lo) / (t_hi - t_lo), 0.0, 1.0)


def apply_neg_log_one_minus(raw, sim_type, p):
    eps = p["eps"]
    v = -np.log(np.clip(1.0 - raw + eps, eps, None))
    denom = p["v_hi"] - p["v_lo"]
    if denom <= 0:
        return np.full_like(v, 0.5)
    return np.clip((v - p["v_lo"]) / denom, 0.0, 1.0)


def apply_power(raw, sim_type, p):
    denom = p["hi"] - p["lo"]
    if denom <= 0:
        return np.full_like(raw, 0.5)
    u = np.clip((raw - p["lo"]) / denom, 0.0, 1.0)
    return u ** p["gamma"]


def apply_exp_l2(raw, sim_type, p):
    return np.exp(-raw / max(p["tau"], 1e-12))


def apply_ecdf(raw, sim_type, p):
    """Empirical-CDF (rank equalization) — the contract-forbidden baseline.
    p['sorted'] holds the sorted oriented calibration pool."""
    x = orient(raw, sim_type)
    s = p["sorted"]
    r = np.searchsorted(s, x.ravel(), side="right") / len(s)
    return r.reshape(x.shape)


def fit_ecdf(pool: np.ndarray, sim_type: str) -> dict:
    return {"sorted": np.sort(orient(pool, sim_type))}


NORMALIZERS = {
    "legacy_percentile": (fit_legacy_percentile, apply_legacy_percentile),
    "zscore": (fit_zscore, apply_zscore_tanh),
    "affine_clip": (fit_affine_clip, apply_affine_clip),
    "logit": (fit_logit, apply_logit),
    "neg_log_one_minus": (fit_neg_log_one_minus, apply_neg_log_one_minus),
    "power": (fit_power, apply_power),
    "exp_l2": (fit_exp_l2, apply_exp_l2),
    "ecdf": (fit_ecdf, apply_ecdf),
}


# ------------------------------------------------------------------
# Squash family on standardized z (expC): all take z -> [0,1] except identity
# ------------------------------------------------------------------
def _ndtr(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0))) if np.isscalar(z) else \
        0.5 * (1.0 + _erf_np(z / math.sqrt(2.0)))


def _erf_np(x: np.ndarray) -> np.ndarray:
    # torch.erf is vectorized and fast; avoids scipy dependency.
    return torch.erf(torch.from_numpy(np.ascontiguousarray(x))).numpy()


SQUASHES = {
    "tanh": lambda z: 0.5 * (np.tanh(z) + 1.0),
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "logistic2z": lambda z: 1.0 / (1.0 + np.exp(-2.0 * z)),  # identity check vs tanh
    "probit": lambda z: _ndtr(z),
    "arctan": lambda z: 0.5 + np.arctan(z) / math.pi,
    "softsign": lambda z: 0.5 * (z / (1.0 + np.abs(z)) + 1.0),
    "hardclip1": lambda z: np.clip(0.5 + z / 2.0, 0.0, 1.0),
    "hardclip2": lambda z: np.clip(0.5 + z / 4.0, 0.0, 1.0),
    "hardclip3": lambda z: np.clip(0.5 + z / 6.0, 0.0, 1.0),
    "identity": lambda z: z,
}


# ------------------------------------------------------------------
# Fusion + retrieval evaluation (LOEO: own trajectory always masked)
# ------------------------------------------------------------------
def fuse(norm_scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    for f, w in weights.items():
        term = w * norm_scores[f]
        out = term if out is None else out + term
    return out


def rank_matrix(S: np.ndarray, same_traj: np.ndarray) -> np.ndarray:
    """Return per-row candidate ordering (best first) with own trajectory masked."""
    S = S.copy()
    S[same_traj] = -np.inf
    return np.argsort(-S, axis=1, kind="stable")


def retrieval_metrics(
    art: Artifact,
    S: np.ndarray,
    k_prec: int = 5,
    k_ndcg: int = 10,
) -> dict[str, np.ndarray]:
    """Per-query metric vectors (aggregate with cluster_bootstrap)."""
    n = art.n
    order = rank_matrix(S, art.same_traj)
    top1 = order[:, 0]
    same_task = art.same_task

    rows = np.arange(n)
    m: dict[str, np.ndarray] = {}
    m["top1_same_task"] = same_task[rows, top1].astype(np.float64)

    topk = order[:, :k_prec]
    m[f"p@{k_prec}_same_task"] = same_task[rows[:, None], topk].mean(axis=1)

    # MRR of the first same-task candidate in the LOEO ranking.
    st_sorted = same_task[rows[:, None], order]
    first = np.argmax(st_sorted, axis=1)
    has = st_sorted.any(axis=1)
    mrr = np.where(has, 1.0 / (first + 1.0), 0.0)
    m["mrr_same_task"] = mrr

    # nDCG@k with binary same-task relevance.
    kk = min(k_ndcg, order.shape[1])
    gains = st_sorted[:, :kk].astype(np.float64)
    disc = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = (gains * disc).sum(axis=1)
    n_rel = np.minimum((same_task & ~art.same_traj).sum(axis=1), kk)
    n_rel = np.maximum(n_rel, 1)
    idcg = np.array([disc[: int(r)].sum() for r in n_rel])
    m[f"ndcg@{k_ndcg}"] = dcg / idcg

    # Action-replay proxy: mean squared distance between the query's own action
    # chunk and the retrieved one; plus oracle/random regret normalization.
    d2 = art.action_d2
    allowed = ~art.same_traj
    m["action_mse@1"] = d2[rows, top1]
    d2_masked = np.where(allowed, d2, np.inf)
    oracle = d2_masked.min(axis=1)
    rand = np.where(allowed, d2, np.nan)
    rand_mean = np.nanmean(rand, axis=1)
    denom = np.maximum(rand_mean - oracle, 1e-12)
    m["action_regret@1"] = (m["action_mse@1"] - oracle) / denom

    # Phase alignment among all queries (phase in [0,1] within a trajectory).
    ph = art.step_idx / np.maximum(art.traj_len - 1, 1)
    m["phase_err@1"] = np.abs(ph - ph[top1])
    return m


def cluster_bootstrap(
    values: np.ndarray, clusters: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% CI by resampling whole trajectories (clusters)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        cs = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[c] for c in cs])
        means[b] = values[sel].mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_cluster_bootstrap(
    a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """CI of mean(a-b) under trajectory-cluster resampling."""
    return cluster_bootstrap(a - b, clusters, n_boot=n_boot, seed=seed)
