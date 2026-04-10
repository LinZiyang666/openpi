# Cache Key Dimensionality Reduction Recommendations

## Current Status Analysis

The current `FullOriginalKeyBuilder` directly flattens all raw embeddings as cache keys:

| Field | Original Shape | Flattened Dimension |
|------|----------|---------------|
| vision_0 (base_0_rgb) | [256, 2048] | 524,288 |
| vision_1 (left_wrist) | [256, 2048] | 524,288 |
| vision_2 (right_wrist) | [256, 2048] | 524,288 |
| prompt_emb | [200, 2048] | 409,600 |
| robot_state | [32] | 32 |
| **Total** | | **~1,982,496** |

Problems:
- Excessively high dimensionality forces Qdrant to chunk (>65k dimensions per chunk), degrading search efficiency
- No L2 normalization, making cosine similarity results unreliable
- Distance metrics degrade in high-dimensional spaces (curse of dimensionality)

---

## Dimensionality Reduction Pipeline

Methods are divided into two layers, applied sequentially:

```
Raw embedding          Layer 1: Token Pooling           Layer 2: Dimension Projection
[256, 2048]  ──────►  [2048] or [K, 2048]  ──────►  [D]
(~500K dim/field)     (2048~32768 dim/field)          (128~512 dim/field)
```

> **Important note:** Layer 2 methods (PCA, random projection, etc.) operate on the 2048-dimensional vector **after** Layer 1 pooling, not directly on the raw 500K-dimensional data. Applying PCA/projection directly to million-dimensional data is neither practical (memory and compute prohibitive) nor necessary — pool first, then reduce is the standard approach.

---

## Layer 1: Token Pooling (Required, 500K → 2048)

### Method A: Mean Pooling

Average the token sequence along the token dimension to get a single vector.

```
[256, 2048] → mean(dim=0) → [2048]
```

| Field | Before Pooling | After Pooling |
|------|--------|--------|
| vision_x (x3) | 524,288 | 2,048 |
| prompt_emb | 409,600 | 2,048 |
| robot_state | 32 | 32 (unchanged) |
| **Total** | **1,982,496** | **8,224** |

- **Compression ratio: 241x**
- SigLIP/CLIP family models' standard image-level representation is mean pooling
- Simplest implementation, suitable as a baseline

### Method B: Spatial Pooling (Vision Only)

Vision tokens are arranged on a 16x16 grid (patch_size=14, image=224). Adaptive average pooling preserves spatial structure:

```
[256, 2048] → reshape [16, 16, 2048] → adaptive_avg_pool2d → [H, W, 2048] → flatten
```

| Pooling Granularity | Token Count | Dim/Field | Compression Ratio |
|----------|----------|-----------|--------|
| 4x4 | 16 | 32,768 | 16x |
| 2x2 | 4 | 8,192 | 64x |
| 1x1 (=Mean) | 1 | 2,048 | 256x |

- Preserves spatial layout information (which region contains which objects)
- Suitable for tasks where scene differences are primarily spatial
- Prompt still uses Mean Pooling

### Method C: Attention-Weighted Pooling

Use CLS token or a learnable query to compute a weighted sum over tokens:

```
weights = softmax(tokens @ query)   # [256, 1]
pooled = (tokens * weights).sum(0)  # [2048]
```

- More targeted than mean pooling, emphasizes important tokens
- Requires training or manually defining a query vector
- Medium implementation complexity

---

## Layer 2: Dimension Projection (Optional, 2048 → 128~512)

> All methods below operate on the **2048-dimensional** vector after Layer 1 pooling.

### Method D: Random Projection

Project using a fixed-seed random Gaussian matrix. The Johnson-Lindenstrauss lemma guarantees approximate distance preservation.

```
proj_matrix = normalize(randn(2048, 256), dim=0)  # fixed seed
reduced = pooled @ proj_matrix                      # [2048] → [256]
```

- **Requires no training data** — just generate the matrix
- Theoretical guarantee: projecting N points to O(log N / epsilon^2) dimensions preserves pairwise distances within epsilon
- Suitable for quick validation

### Method E: PCA

Principal component analysis on the pooled vector collection.

```
pca = PCA(n_components=256).fit(pooled_samples)  # offline fitting
reduced = pca.transform(pooled)                    # [2048] → [256]
```

- Requires pre-collecting a batch of samples (hundreds to thousands) for fitting
- Must save the transformation matrix for inference-time use
- Typically outperforms random projection

### Method F: Learned Projection Head

Train a small MLP for dimensionality reduction, optimized with contrastive loss.

```
class ProjectionHead(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
    def forward(self, x):
        return F.normalize(self.net(x))
```

- Best approach when optimized for a specific task domain
- Requires training data + training pipeline
- Highest implementation and maintenance cost

---

## Recommended Implementation Path

| Phase | Combination | Dim/Field | Total Dim | Compression Ratio |
|------|------|-----------|--------|--------|
| **Phase 1** | Mean Pooling + L2 Norm | 2,048 | 8,224 | 241x |
| **Phase 2** | Mean Pooling + Random Projection (256d) | 256 | 1,060 | 1,871x |
| **Phase 3** (optional) | Mean Pooling + Learned Projection (128d) | 128 | 544 | 3,644x |

Phase 1 has the highest priority because:
1. Compression ratio is already enormous (241x); Qdrant no longer needs chunking
2. Mean Pooling is the standard approach for SigLIP retrieval, with virtually no semantic loss
3. Fixes the current missing L2 normalization problem
4. Simplest implementation, achievable within the existing `QueryKeyBuilder` Protocol

Phase 2 can be added after Phase 1 validation as needed, further reducing storage and search overhead.

Phase 3 should only be considered if Phase 2 results are insufficient.

---

## Implementation Approach

Within the existing architecture, create a new `MeanPoolKeyBuilder` (or parameterized `PooledKeyBuilder`) implementing the `QueryKeyBuilder` Protocol, switchable via `cache.yaml`'s `key_builder.type`. No modifications needed to Orchestrator, SearchStrategy, or Backend code.

---
---

# Similarity Computation Recommendations

## Problem Decomposition

Similarity computation naturally decomposes into two layers:

```
              Layer 1: Field Similarity              Layer 2: Cross-Modal Fusion
query_key ──────────────────────────► per-field score ──────────────────────────► final score
(per field)    "how similar are two same-modality vectors"  (per field)  "how to merge scores from multiple modalities"
```

| Layer | Responsibility | Current Implementation |
|----|------|----------|
| **Layer 1: Field Similarity** | Compare two vectors of the same field (e.g., vision_0) | Cosine Similarity (Qdrant native / InMemory F.cosine_similarity) |
| **Layer 2: Cross-Modal Fusion** | Merge similarities from multiple fields into a final ranking/score | Qdrant Weighted RRF; InMemory only uses the first field |

The two layers are independent design decisions and can be freely combined.

---

## Layer 1: Field Similarity (Vector Comparison Within a Single Field)

### Method 1A: Cosine Similarity (Current Approach)

```
sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

- Range [-1, 1], 1 = identical direction
- **Insensitive to vector magnitude**, only considers direction
- Standard choice for embedding retrieval (CLIP, SigLIP, sentence-transformers all use this)
- If keys are already L2-normalized, cosine similarity = dot product, which is faster to compute

**Applicable scenario:** Default choice for most cases. SigLIP embeddings are inherently directional in semantics; cosine is the most appropriate metric.

### Method 1B: L2 Distance (Euclidean Distance)

```
dist(a, b) = ‖a - b‖₂
sim = -dist  or  sim = 1 / (1 + dist)
```

- Sensitive to vector magnitude — two vectors with the same direction but different lengths have non-zero distance
- On L2-normalized vectors, L2 distance and cosine similarity are monotonically equivalent: `‖a-b‖² = 2(1 - cos(a,b))`
- Natively supported by Qdrant (`Distance.EUCLID`)

**Applicable scenario:** robot_state field. Joint angles / end-effector positions have meaningful absolute values and should not be compared by direction alone. For example, `[0.1, 0.2]` and `[0.2, 0.4]` have the same direction but different states; L2 can distinguish them.

### Method 1C: Dot Product

```
sim(a, b) = a · b
```

- If vectors are L2-normalized, equivalent to cosine similarity
- If not normalized, considers both direction and magnitude (larger magnitude = higher confidence embeddings contribute more)
- Fastest distance metric in Qdrant (no norm computation needed)

**Applicable scenario:** When embedding magnitude itself carries information (e.g., some transformer layer outputs where magnitude correlates with token importance). However, this assumption needs verification for SigLIP prefix_embs.

### Method 1D: Learned Metric

Train a small network to judge similarity between two vectors:

```python
# Siamese approach
sim = mlp(concat(a, b))  # or mlp(|a - b|)

# Mahalanobis approach (learn a weighting matrix M)
sim = -sqrt((a-b)ᵀ M (a-b))
```

- Can learn non-linear similarity definitions
- Mahalanobis is a generalization of L2: M=I degenerates to L2, M=diagonal = weighted L2
- Training is similar to the dimensionality reduction MLP (contrastive learning + positive/negative pairs)

**Applicable scenario:** When simple metrics are insufficient. In practice, if the dimensionality reduction stage uses a Learned Projection, the projection has already implicitly learned a metric (cosine in the projected space = learned metric in the original space), so this layer usually does not need additional learning.

### Layer 1 Recommendation

| Field | Recommended Metric | Rationale |
|------|---------|------|
| vision_0/1/2 | **Cosine** | SigLIP embedding semantics are in the direction; magnitude is meaningless |
| prompt_emb | **Cosine** | Same as above; standard practice for language embeddings |
| robot_state | **L2 Distance** | Joint angles/positions have physically meaningful absolute values |

> **Key insight:** Different fields can use different metrics, and this is reasonable. Qdrant supports independent distance metric configuration per named vector. The current implementation uses cosine uniformly for all fields, which is not ideal for robot_state.

---

## Layer 2: Cross-Modal Fusion (Multi-Field Score Merging)

### Method 2A: Weighted RRF (Current Approach)

```
RRF_score(d) = Σ_field  w_field / (k + rank_field(d))
```

- Based on rank rather than raw scores; naturally compatible with different metrics/scales
- k parameter controls weight concentration on top ranks (smaller k = more concentrated on top-1)
- Current k=60, fusion_weights all 1.0

**Advantages:**
- No score normalization needed; different metrics can be fused directly
- Rank fusion is more robust to noise than score fusion
- Natively supported by Qdrant

**Disadvantages:**
- Loses absolute score magnitude information (two results with cosine=0.99 and cosine=0.50 may differ by only 1 in rank)
- Optimal k and weights require experimental tuning
- Only supported by Qdrant; InMemory backend does no multi-field fusion at all

### Method 2B: Weighted Score Sum

```
final_score(d) = Σ_field  w_field · sim_field(q, d)
```

The most intuitive method — each field's similarity multiplied by weight, then summed.

**Prerequisite:** All field scores must be on the same scale. If vision uses cosine in [-1,1] while robot_state uses L2 in [0, +infinity), direct weighted summation is meaningless. Normalization to a unified range is needed first.

**Normalization methods:**
- Min-max: `(score - min) / (max - min)` → [0, 1] (requires global or batch statistics)
- Z-score: `(score - mu) / sigma` (requires historical statistics)
- Fixed range mapping: cosine is already in [-1,1]; L2 mapped via `1/(1+dist)` to (0,1]

**Advantages:**
- Preserves absolute score value information
- Easy to understand and debug
- Implementable in any backend (not Qdrant-RRF dependent)

**Disadvantages:**
- Requires score normalization; normalization strategy matters significantly when mixing different metrics
- Anomalous scores (e.g., one field is all low scores) can skew the total

### Method 2C: Learned Fusion

Train a small model that takes per-field similarity scores as input and outputs a final score.

```python
# Input: [sim_vision_0, sim_vision_1, sim_vision_2, sim_prompt, sim_state]
# Output: final_score ∈ [0, 1]

fusion_net = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
)
```

**Training signal:** Requires labeling "whether these two steps should be considered similar," which can come from:
- Adjacent steps within the same episode → positive sample (should score high)
- Random steps across episodes → negative sample (should score low)
- Corresponding stages of different episodes in the same task → positive sample

**Advantages:**
- Automatically learns optimal weights and non-linear combinations across fields
- Can learn interaction rules like "vision matches but state doesn't → low score"

**Disadvantages:**
- Requires offline training + deployment
- A 5-dimensional input MLP is roughly equivalent to hand-tuned weights
- Cannot leverage Qdrant's native fusion acceleration; requires fetching top-K per field then local re-ranking

### Method 2D: Two-Stage Retrieval

```
Stage 1 (Recall): Single-field fast retrieval of top-100 candidates
Stage 2 (Re-rank): Compute full multi-field weighted scores on candidate set, re-rank for top-K
```

- Stage 1 uses the most discriminative single field (typically vision_0) for fast recall
- Stage 2 performs fine-grained multi-field fusion on the small candidate set

**Advantages:**
- Balances speed and accuracy
- Stage 2 can use arbitrarily complex fusion methods (including learned fusion)
- Reduces overhead of multi-field parallel search

**Disadvantages:**
- Poor field choice in Stage 1 may miss good results
- Higher implementation complexity

### Layer 2 Recommendation

| Method | Applicable Phase | Rationale |
|------|---------|------|
| **Weighted Score Sum** | Phase 1 primary | Simple and intuitive, preserves score information, implementable in all backends |
| **Weighted RRF** | Phase 1 alternative | Already implemented; rank fusion naturally compatible with different metrics |
| **Two-Stage** | Phase 2 optimization | Improves speed at scale while maintaining accuracy |
| **Learned Fusion** | Phase 3 | Mostly equivalent to hand-tuned weights in practice; low priority |

> **Key insight:** If Layer 1 uses cosine for all fields (normalized scores are comparable), Weighted Score Sum is better than RRF — it preserves the information that "this vision match is 0.99 vs 0.60," whereas RRF only sees rank. RRF's advantage scenario is when field metrics are non-uniform (e.g., vision uses cosine, state uses L2).

---

## Combined Recommendations

| Phase | Layer 1 (Field Similarity) | Layer 2 (Fusion) |
|------|---------------------------|-------------------|
| **Phase 1** | vision/prompt: Cosine, state: Cosine | Weighted Score Sum (equal weights first, tune experimentally later) |
| **Phase 1-alt** | All Cosine | Keep Weighted RRF (already implemented; verify dimensionality reduction effect first) |
| **Phase 2** | vision/prompt: Cosine, state: L2 (normalized to [0,1]) | Weighted Score Sum + experimentally tuned weights |
| **Phase 3** | Same as Phase 2 | Two-Stage (vision recall + full-field re-rank) |

The core Phase 1 recommendation is: **do not change Layer 2 yet; focus on dimensionality reduction (Mean Pooling from the previous section).** The benefit from dimensionality reduction far exceeds that of switching fusion methods. After dimensionality reduction is in place, use A/B experiments to compare RRF vs Score Sum.

---

## Mapping to Existing Architecture

```
Layer 1 (Field Similarity)  →  VectorStoreBackend's distance metric configuration
                                (Qdrant: per-vector distance; InMemory: F.cosine_similarity)

Layer 2 (Cross-Modal Fusion) → SearchStrategy's responsibility
                                (Current: QdrantWeightedRrfKnnStrategy passes RRF params via backend_hints)
                                (Improvement: can do score-level fusion at the SearchStrategy layer)
```

Modifying Layer 1 requires changing Backend configuration (Qdrant collection schema's distance type).
Modifying Layer 2 can be done at the SearchStrategy layer without changing the Backend — retrieve per-field top-K, re-rank within the Strategy.

---
---

# Dual Encoder vs Cross Encoder: Learned Similarity Architectures

## Concepts and Positioning

Dual Encoder and Cross Encoder are two core architectures in information retrieval, corresponding to the **recall** and **re-ranking** stages of a retrieval system respectively. They are not merely "field-level learned similarity" but a fundamental rethinking of the entire similarity pipeline.

```
                    Dual Encoder                    Cross Encoder
                    (Bi-Encoder)                    (Reranker)

Input method:  query and candidate encoded independently  query and candidate concatenated for joint encoding
Output:        individual embeddings → comparison         directly outputs similarity score
Pre-computation: ✅ candidates can be encoded offline      ❌ each pair requires real-time computation
Speed:         O(1) per candidate (vector retrieval)      O(n) per candidate (forward pass)
Accuracy:      Medium (no cross-attention)                High (full cross-attention)
Typical use:   Stage 1 recall top-K                       Stage 2 re-rank K candidates
```

Mapping to our cache system:

```
Current architecture:  KeyBuilder(frozen SigLIP) + cosine  ≈  frozen Dual Encoder

Improvement 1:  Trained Dual Encoder           →  replaces KeyBuilder + dimensionality reduction layer
Improvement 2:  Dual Encoder recall + Cross Encoder re-rank  →  Two-Stage architecture
```

---

## Dual Encoder (Bi-Encoder)

### Principle

Query and candidate are each mapped to a shared embedding space through an encoder, then compared using a simple metric (cosine/dot product). The two encoders can share parameters (Siamese) or be independent (Asymmetric).

```
query step:      [vision, prompt, state] → Encoder_Q → z_q ∈ ℝ^d
candidate step:  [vision, prompt, state] → Encoder_C → z_c ∈ ℝ^d

similarity = cosine(z_q, z_c)  or  z_q · z_c
```

### Relationship to Current Architecture

The current system is effectively already a dual encoder, except the encoder is frozen:

```
Current:   SigLIP(frozen) → flatten/pool → cosine
           encoder is not optimized for "similar step retrieval" task

Trained:   SigLIP(frozen) → Projection MLP(trained) → cosine
           MLP learns what kind of steps should be considered similar
```

The previously discussed Learned Projection Head (Method F in Layer 2 dimensionality reduction) is essentially training the projection layer of a dual encoder.

### How to Apply in Our Scenario

**Architecture choice: Add projection head on top of frozen backbone**

Do not fine-tune SigLIP/PaliGemma (too large, would degrade inference quality); only train the projection layer on top.

```python
class DualEncoderHead(nn.Module):
    """Project multi-modal stage1 outputs to a unified retrieval space"""
    def __init__(self, vision_dim=2048, state_dim=32, out_dim=256):
        super().__init__()
        # Independent projection per modality
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))
        self.prompt_proj = nn.Sequential(
            nn.Linear(vision_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))
        # Cross-modal fusion (optional)
        self.fusion = nn.Linear(out_dim * 5, out_dim)  # 3 vision + prompt + state

    def forward(self, vision_pools, prompt_pool, state):
        """
        vision_pools: list of 3 × [B, 2048] (mean-pooled per image)
        prompt_pool:  [B, 2048] (mean-pooled)
        state:        [B, 32]
        """
        v_embs = [self.vision_proj(v) for v in vision_pools]    # 3 × [B, out_dim]
        p_emb = self.prompt_proj(prompt_pool)                    # [B, out_dim]
        s_emb = self.state_proj(state)                           # [B, out_dim]

        fused = self.fusion(torch.cat(v_embs + [p_emb, s_emb], dim=-1))
        return F.normalize(fused, dim=-1)                        # [B, out_dim]
```

**Key design decision: Per-field independent encoding or fused single vector?**

| Approach | Advantages | Disadvantages |
|------|------|------|
| **Per-field independent projection** (each field outputs one vector) | Preserves per-field weight tunability; compatible with Qdrant named vectors | Still needs Layer 2 fusion; inter-field interactions are ignored |
| **Fused single vector** (all fields merged into one vector) | One cosine comparison covers everything; naturally learns cross-modal interactions | Loses per-field interpretability; weights not tunable |

**Recommendation: Start with per-field independent projection** (consistent with dimensionality reduction MLP); fusion layer can be added later.

### Training Method

Identical to the previously discussed Learned Projection Head:

1. **Data:** Episode data collected in HDF5, mean pooled to get per-step features
2. **Positive pairs:** Adjacent steps within the same episode (time window ±3)
3. **Negatives:** In-batch negatives (other steps within the same batch)
4. **Loss function:** InfoNCE

```python
# query, candidate: [B, out_dim]  (L2 normalized)
sim_matrix = query @ candidate.T / temperature  # [B, B]
labels = torch.arange(B)
loss = F.cross_entropy(sim_matrix, labels)
```

5. **Only train the projection layer**, SigLIP backbone frozen

---

## Cross Encoder (Reranker)

### Principle

The raw features of query and candidate are **concatenated and jointly encoded**, using cross-attention mechanisms to capture fine-grained interactions between them, directly outputting a similarity score.

```
Input: concat([query_features, [SEP], candidate_features])
      → Transformer / MLP
      → scalar score ∈ [0, 1]
```

### Why More Accurate Than Dual Encoder

In a Dual Encoder, query and candidate are encoded independently — there is no information exchange between them. This means the encoder must compress **all potentially needed comparison information** into a fixed-dimensional vector.

A Cross Encoder allows each query token to directly attend to each candidate token, enabling:
- "Query's left wrist image has a red block" + "candidate's left wrist image also has a red block" → high score
- "Query's state shows gripper open" + "candidate's state shows gripper closed" → even if vision is very similar, give low score

This kind of **conditional judgment** is impossible for a dual encoder (independent encoding + cosine).

### How to Apply in Our Scenario

**Approach: Lightweight Cross Encoder as reranker**

No need for a large Transformer — input dimensionality is already small (after mean pooling, each step is only ~8k dims); an MLP suffices.

```python
class CrossEncoderReranker(nn.Module):
    """Re-rank top-K candidates recalled by Dual Encoder"""
    def __init__(self, per_step_dim=8224):
        super().__init__()
        # Input: query features + candidate features + element-wise interaction features
        input_dim = per_step_dim * 3  # [q, c, |q-c|]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_feats, candidate_feats):
        """
        query_feats:     [B, D]  (one query repeated K times)
        candidate_feats: [B, D]  (K candidates)
        returns:         [B, 1]  similarity scores
        """
        interaction = torch.abs(query_feats - candidate_feats)
        combined = torch.cat([query_feats, candidate_feats, interaction], dim=-1)
        return self.net(combined)
```

**Input feature design:**

| Feature | Dimension | Meaning |
|------|------|------|
| `query_feats` | D | Pooled features of the query step |
| `candidate_feats` | D | Pooled features of the candidate step |
| `\|query - candidate\|` | D | Element-wise difference (which dimensions differ) |
| **Total input** | **3D** | Concatenated and fed into MLP |

The element-wise difference `|q-c|` is critical — it directly encodes "what is different," which is more efficient than letting the MLP learn this from the `[q, c]` concatenation alone.

### Training Method

**Unlike Dual Encoder — requires pointwise/pairwise labels, not contrastive.**

**Approach A: Pointwise (Binary Classification)**

```
Positive: (query, candidate) from adjacent steps in same episode → label = 1
Negative: (query, candidate) from random steps across episodes → label = 0
Loss:     BCE loss
```

**Approach B: Pairwise (Learning to Rank)**

```
Triplet: (query, positive, negative)
  positive = adjacent step from same episode
  negative = random step

Loss: margin ranking loss
  loss = max(0, margin - score(q, pos) + score(q, neg))
```

**Approach C: Distillation from Oracle (Recommended)**

If a "ground truth similarity" definition exists (e.g., L2 distance between two steps' action chunks), it can be used as a soft label:

```python
# oracle_sim: "true" similarity computed from action chunk distance
# pred_sim: cross encoder's predicted similarity
loss = F.mse_loss(pred_sim, oracle_sim)
```

This is more informative than binary labels — it learns not just "similar/not similar" but "how similar."

### Training Data Volume Estimate

Cross Encoder requires more training data than Dual Encoder (because in-batch negatives cannot be used; explicit positive/negative pairs must be constructed):

| Data Volume | Effect |
|--------|------|
| 1k pairs | Basically usable, prone to overfitting |
| 10k pairs | Reasonable performance |
| 100k+ pairs | Stable generalization |

Generating pairs from HDF5 episode data is easy — each episode has hundreds of steps, providing large combinatorial volumes.

---

## Complete Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Query Step                          │
│  [vision_0, vision_1, vision_2, prompt_emb, state]     │
│                        │                                │
│                   Mean Pool                             │
│                        │                                │
│              ┌─────────┴─────────┐                      │
│              ▼                   ▼                      │
│     Dual Encoder Head    (raw pooled features           │
│     → query_emb [256]     kept for Stage 2)             │
│              │                                          │
│              ▼                                          │
│  ┌───── Stage 1: Recall ───────┐                        │
│  │  Qdrant cosine search       │                        │
│  │  query_emb vs all           │                        │
│  │  → top-K candidates         │                        │
│  └───────────┬─────────────────┘                        │
│              │ K candidates (IDs + pooled features)     │
│              ▼                                          │
│  ┌───── Stage 2: Re-rank ──────┐                        │
│  │  Cross Encoder MLP          │                        │
│  │  input: [q, c, |q-c|]      │                        │
│  │  → re-scored top-K          │                        │
│  └───────────┬─────────────────┘                        │
│              │                                          │
│              ▼                                          │
│       Final ranked result                               │
└─────────────────────────────────────────────────────────┘
```

### Latency Budget Analysis

| Stage | Operation | Estimated Latency |
|------|------|---------|
| Mean Pool | 3 means + L2 norm | < 0.1 ms |
| Dual Encoder Head | MLP forward (5 Linear layers) | < 0.5 ms |
| Stage 1: Qdrant search | Single-vector cosine top-50 | 1-5 ms |
| Stage 2: Cross Encoder | 50 MLP forwards (batchable) | < 1 ms |
| **Total** | | **< 7 ms** |

Compared to current: 5 named vectors x chunked RRF search ≈ 10-50 ms (depending on data volume and chunk count)

The Two-Stage architecture may actually be **faster**, since Stage 1 only searches one vector.

---

## Dual Encoder vs Cross Encoder Comparison Summary

| Dimension | Dual Encoder | Cross Encoder |
|------|-------------|---------------|
| **Role** | Recall (full-corpus retrieval) | Re-ranking (small candidate set re-ordering) |
| **Input** | Query and candidate encoded independently | Query + candidate jointly encoded |
| **Speed** | Fast (pre-computation + ANN index) | Slow (real-time computation per pair) |
| **Accuracy** | Medium (no cross-modal interaction) | High (full cross-attention) |
| **Training data** | Less (in-batch negatives) | More (explicit positive/negative pairs) |
| **Pre-computable** | ✅ Candidates encoded offline into index | ❌ Must recompute for each query |
| **In our system** | Replaces KeyBuilder + dimensionality reduction layer | New reranker, within SearchStrategy |

---

## Implementation Recommendations

| Phase | What to Do | Rationale |
|-------|--------|------|
| **Phase 1** | Mean Pool + L2 Norm (no learning) | Eliminate dimensionality issues first, establish baseline |
| **Phase 2** | Train Dual Encoder Head | Optimize embedding space for "step similarity" |
| **Phase 3** | Add Cross Encoder Reranker | Improve accuracy, especially for borderline cases |

Phase 2 and Phase 3 can be trained and deployed independently. Deploy the Dual Encoder first, accumulate retrieval data for a period, then use that data to train the Cross Encoder.

### Integration Points with Existing Architecture

```
Dual Encoder Head   →  New KeyBuilder implementation (DualEncoderKeyBuilder)
                       build() calls MLP forward internally, outputs single or per-field vectors

Cross Encoder       →  New SearchStrategy implementation (TwoStageStrategy)
                       search() internally:
                         1. Calls storage.search() to get top-K (Stage 1)
                         2. Fetches candidate features
                         3. Cross encoder re-rank (Stage 2)
                         4. Returns re-ranked results
```

No modifications needed to Orchestrator, Gate, Judge, or Backend.

---

## CP1 Calibration Experiment Findings and Improvement Directions (2026-04-06)

### Calibration Results

After building artifacts (1040 entries) with 4 pooling methods (mean_pool / spatial_pool_16 / spatial_pool_64 / max_pool), running `calibrate_score_sum_stats.py` to compute same-task vs cross-task discriminability (separation = same_task_mean - cross_task_mean):

| Field | Similarity Type | Separation Range | Assessment |
|------|-----------|----------------|------|
| vision_0 | cosine | 0.001 ~ 0.003 | Very low; virtually no discriminability after pooling |
| vision_1 | cosine | 0.001 ~ 0.002 | Same as above |
| prompt_emb | cosine | ~0.000002 | Nearly zero; CP1-level prompt embeddings are virtually identical across all tasks |
| robot_state | L2 | ~0.06 | **Best**, 20-60x higher than vision |

**Core problem**: Of the 256 ViT patch tokens, the vast majority correspond to static background (desk, walls); only a few patches capture task-relevant objects (robot arm, target objects). Simple pooling (mean/max/spatial) treats all patches equally, drowning useful signals in background noise.

Relatively speaking, spatial_pool_16 preserves the most spatial information (sep=0.0025), while mean_pool performs worst (sep=0.001).

### Improvement Directions: Smarter Token Processing

#### Direction 1: Variance-based Token Selection (Static Analysis, Simplest)

Pre-compute variance of each token position from existing data, select the top-k most active tokens:

```python
# Compute statistics across all entries' vision_0 [256, 2048]
all_tokens = torch.stack(...)  # [N, 256, 2048]
variance = all_tokens.var(dim=0)  # [256, 2048]
position_importance = variance.sum(dim=1)  # [256]
top_k_positions = position_importance.topk(k=32).indices  # select 32 most active tokens
```

Use only these 32 tokens flattened to a 32x2048 = 65536d vector for cosine comparison.

- **Advantages**: Compute mask once offline, use directly during search, minimal code changes
- **Disadvantages**: Mask is global; different tasks may have different critical tokens

#### Direction 2: MaxSim (ColBERT-style, No Compression)

Skip pooling entirely; perform token-level matching directly:

```python
def maxsim(query, candidate):
    # query: [256, 2048], candidate: [256, 2048]
    sim_matrix = query @ candidate.T  # [256, 256]
    # For each query token, find the most similar candidate token
    max_per_query = sim_matrix.max(dim=1).values  # [256]
    return max_per_query.mean()  # scalar
```

A method validated in information retrieval (ColBERT).

- **Advantages**: Preserves full token-level matching, no information loss; theoretically highest discriminability
- **Disadvantages**: High computation cost (each comparison is a 256x256 matrix multiplication); but for 1040 entries batched, this should be acceptable

#### Direction 3: Residual Encoding (Subtract Mean Template)

```python
# Offline: compute mean template across all entries
mean_template = all_tokens.mean(dim=0)  # [256, 2048]

# Each entry's key = tokens - mean_template (removes shared static content)
residual = tokens - mean_template
key = residual.flatten()  # or further pool the residual
```

- **Core idea**: Shared content across all entries (background) is subtracted, leaving only the differential parts
- **Advantages**: Cosine discriminability after pooling will significantly improve; minimal code changes
- **Disadvantages**: Depends on the representativeness of the mean template

#### Direction 4: Cross-step Attention (Temporal Difference Analysis)

Identify dynamic tokens using within-episode temporal token changes:

```python
# Compare consecutive steps within the same episode
delta = tokens_t - tokens_{t-1}  # [256, 2048]
dynamic_mask = delta.norm(dim=1) > threshold  # which tokens changed significantly
```

Tokens with large changes correspond to robot arm motion trajectories and object movement — the most discriminative signals.

- **Advantages**: Captures truly dynamic signals
- **Disadvantages**: Requires temporal information; more complex artifact construction

### Recommended Implementation Order

| Priority | Direction | Rationale |
|--------|------|------|
| **P0** | Direction 3 (Residual) + Direction 1 (Variance Selection) combined | Low cost, minimal changes; only need to pre-compute mean_template and variance mask during artifact construction; add a new `CP1ResidualKeyBuilder`; fully compatible with existing experiment framework |
| **P1** | Direction 2 (MaxSim) | Potentially best results, but requires modifying similarity computation logic (no longer simple vector cosine); needs a new similarity type in `in_memory_backend.py` |
| **P2** | Direction 4 (Cross-step) | Requires more data processing logic, but can be combined with Direction 1 |

These directions are orthogonal to the Dual Encoder / Cross Encoder approaches described earlier in this document — the former improves token representation/matching (signal extraction), the latter improves the overall embedding space (learned optimization). Both can be stacked.
