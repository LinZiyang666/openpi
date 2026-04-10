====================================================================
  Pi0.5 Cache System - Complete Workflow Diagram
====================================================================


1. System Overview: Startup Flow
====================================================================

  serve_policy.py --cache_config cache.yaml
       |
       v
  load_cache_config(yaml)  -->  validate_cache_config()  -->  CacheConfig
       |
       v
  build_cache_components(config)
       |
       |-- SystemTimer
       |-- VectorStoreBackend (Qdrant / InMemory)
       |-- CacheStorage (facade)
       |-- QueryKeyBuilder (placeholder / cp1_mean_pool / ...)
       |-- Gate (AlwaysSearch)
       |-- Judge (Threshold / AlwaysHit)
       |-- SearchStrategy (WeightedRrfKnn / WeightedScoreSum / QdrantRrf)
       +-- WritePolicy (OnAnyMiss / Always / Never)
                |
                v
       CacheOrchestrator(storage, key_builder, gates, judges, strategies)
                |
                v
       InferenceInterceptor(policy, timer, orchestrator)
                |
                v
       WebsocketPolicyServer(interceptor)  <-- external interface unchanged


2. Single Inference (infer) Complete Flow
====================================================================

Client sends observation
         |
         v
+-- InferenceInterceptor.infer(obs) ----------------------------------+
|                                                                      |
|  (1) Input Transforms (identical to original Policy.infer)           |
|      obs -> input_transform() -> torch tensors -> Observation        |
|                                                                      |
|  (2) Stage 1: Vision + Token Preparation                             |
|      +--------------------------------------+                        |
|      | SigLIP vision encoder (~400M params) |                        |
|      | Prompt tokenization                  |                        |
|      | State discretization                 |                        |
|      | -> prefix_tokens, KV cache           |                        |
|      +------------------+-------------------+                        |
|                         | stage1 output                              |
|                         v                                            |
|  +-- CP1 Cache Check ----------------------------------------+      |
|  |                                                            |      |
|  |  Orchestrator.check(CP1, stage1=stage1)                    |      |
|  |       |                                                    |      |
|  |       v                                                    |      |
|  |  [collect] KeyBuilder.collect(CP1, stage1)                 |      |
|  |       |    extract vision embedding / robot_state / prompt |      |
|  |       v                                                    |      |
|  |  [gate] GateFunction(CP1, cached_data) -> SEARCH / SKIP   |      |
|  |       |                                                    |      |
|  |       v                                                    |      |
|  |  [build] KeyBuilder.build(CP1)                             |      |
|  |       |  -> query_keys: dict[str, Tensor]                  |      |
|  |       |    e.g. {vision_0: [2048], robot_state: [32]}      |      |
|  |       |                                                    |      |
|  |  +----+---- Gate Result Branch ------------------------+   |      |
|  |  |                                                     |   |      |
|  |  | SKIP:                    SEARCH:                    |   |      |
|  |  | return MISS              |                          |   |      |
|  |  | (query_keys populated)   v                          |   |      |
|  |  |                   [search] Strategy.search(ctx)     |   |      |
|  |  |                          | vector retrieval top-k   |   |      |
|  |  |                          v                          |   |      |
|  |  |                   [judge] Judge(results)            |   |      |
|  |  |                          |                          |   |      |
|  |  |                   +------+------+                   |   |      |
|  |  |                   |             |                   |   |      |
|  |  |                 MISS        FULL_HIT                |   |      |
|  |  |                   |             |                   |   |      |
|  |  |                   |        [fetch]                  |   |      |
|  |  |                   |        fetch_payload()          |   |      |
|  |  |                   |        retrieve action_chunk    |   |      |
|  |  +-------------------+-------------+------------------+   |      |
|  +------------------------------------------------------------+      |
|                         |                                            |
|              +----------+----------+                                 |
|              |                     |                                 |
|          FULL_HIT                MISS                                |
|              |                     |                                 |
|              v                     v                                 |
|  +----------------+    (3) Stage 2: LLM Backbone                    |
|  | Skip S2+S3     |       +------------------------------+          |
|  | Use cached     |       | Gemma 2B PaliGemma (~2B)     |          |
|  | action         |       | Prefix-LM attention          |          |
|  |                |       | Fill KV cache (non-autoregr.) |          |
|  | broadcast_     |       +---------------+--------------+          |
|  | action()       |                       | stage2 output            |
|  |                |                       v                          |
|  | buffer_for_    |    (4) Stage 3: Action Expert (Flow Matching)    |
|  | write()        |       +------------------------------+          |
|  |                |       | Gemma 300M + adaRMSNorm      |          |
|  | return cached  |       | 10-step Euler ODE denoising  |          |
|  | action         |       | x1 -> x0                     |          |
|  +-------+--------+       | -> action_chunk [50, 32]     |          |
|          |                +---------------+--------------+          |
|          |                                | stage3 output            |
|          |                                v                          |
|          |                 +-- CP3 Cache Check ----------------+     |
|          |                 | Orchestrator.check(CP3, stage1,   |     |
|          |                 |                    stage3)         |     |
|          |                 | (infra validation, does not affect |     |
|          |                 |  current output)                   |     |
|          |                 +-----------------------------------+     |
|          |                                |                          |
|          |                                v                          |
|          |                 broadcast_action(action_chunk)             |
|          |                 buffer_for_write(query_keys, action_chunk) |
|          |                 orchestrator.clear()                       |
|          |                                |                          |
|          v                                v                          |
|  (5) Output Transforms                                              |
|      outputs -> output_transform() -> numpy -> return to Client      |
+----------------------------------------------------------------------+


3. Episode Lifecycle and Write Flow
====================================================================

WebSocket connection established
       |
       v
  on_task_begin()
  |-- Timer.on_task_begin()
  +-- Orchestrator.on_task_begin()
            |
            v
  +-- Episode Loop -------------------------------------------+
  |                                                            |
  |  on_episode_start(task, episode_id)                        |
  |  |-- step_counter = 0                                      |
  |  |-- clear episode_steps buffer                            |
  |  +-- broadcast to Strategy/Gate/Judge: on_episode_start()  |
  |                                                            |
  |  +-- Step Loop (inference steps) -------------------+      |
  |  |                                                  |      |
  |  |  infer(obs) <-- see flow above                   |      |
  |  |       |                                          |      |
  |  |       |-- CP1 check (step_counter++)             |      |
  |  |       |-- inference or cache hit                 |      |
  |  |       |-- broadcast_action(action)               |      |
  |  |       |   +-- notify all components to record    |      |
  |  |       |       action into trajectory history     |      |
  |  |       |-- buffer_for_write(query_keys, action)   |      |
  |  |       |   +-- append StepRecord to episode_steps |      |
  |  |       +-- orchestrator.clear()                   |      |
  |  |                                                  |      |
  |  +--------------------------------------------------+      |
  |                                                            |
  |  on_episode_end(success)                                   |
  |       |                                                    |
  |       v                                                    |
  |  +-- Write Decision --------------------------------+      |
  |  |                                                   |      |
  |  |  EpisodeRecord = {steps, task_key, miss_counts}   |      |
  |  |       |                                           |      |
  |  |       v                                           |      |
  |  |  WritePolicy.should_write(record)                 |      |
  |  |       |                                           |      |
  |  |  +----+----+                                      |      |
  |  |  |         |                                      |      |
  |  |  No       Yes                                     |      |
  |  |  |         |                                      |      |
  |  |  clear     build_entry_chain(record)              |      |
  |  |  buffer    | each step -> CacheEntry              |      |
  |  |            | linked list (prev_ids / next_ids)    |      |
  |  |            | unique trajectory_id                 |      |
  |  |            v                                      |      |
  |  |       CacheStorage.batch_insert(entries)          |      |
  |  |            |                                      |      |
  |  |            v                                      |      |
  |  |       VectorStoreBackend.batch_insert()           |      |
  |  |       (Qdrant upsert / InMemory append)           |      |
  |  +---------------------------------------------------+      |
  |                                                            |
  +-- next Episode --------------------------------------------+
       |
       v
  on_task_end()
  +-- Timer.on_task_end() -> print per-probe summary


4. Storage Layer Architecture
====================================================================

CacheOrchestrator
       |
       | (sole consumer)
       v
+-- CacheStorage (Facade) ----------------------------------+
|                                                            |
|  Responsibilities:                                         |
|  * Dimension validation (query_keys.shape vs vector_dims)  |
|  * Filter capability check (fail-fast)                     |
|  * Two-phase retrieval:                                    |
|      search() -> SearchResultLite (score only)             |
|      fetch_payload() -> CachePayload (full tensors)        |
|  * Entry integrity validation                              |
|                                                            |
|       |                                                    |
|       v                                                    |
|  VectorStoreBackend (ABC)                                  |
|  +--------------------+   +----------------------------+   |
|  | QdrantVectorStore  |   | InMemoryBackend            |   |
|  | * HTTP/gRPC        |   | * brute-force cosine       |   |
|  | * named vectors    |   | * preload from .pkl        |   |
|  | * step filter      |   | * per-field search         |   |
|  +--------------------+   | * trajectory support       |   |
|                            +----------------------------+   |
+------------------------------------------------------------+


5. Pluggable Components and YAML Config Mapping
====================================================================

cache.yaml                            Component Instance
----------                            ------------------
keys:
  vision_0: {enabled: true}      -->  KeyBuilder: which fields to extract
  robot_state: {weight: 1.0}    -->  SearchStrategy: fusion_weights

key_builder:
  type: cp1_mean_pool            -->  CP1MeanPoolKeyBuilder

checkpoints:
  cp1:
    gate:
      type: always_search        -->  AlwaysSearchGate
    judge:
      type: threshold             -->  ThresholdJudge(threshold=0.98)
      threshold: 0.98
    search_strategy:
      type: weighted_rrf_knn     -->  WeightedRrfKnnStrategy
      top_k: 1
      trajectory_depth: 3        -->  Trajectory-aware search (last 3 steps)
      trajectory_weights:
        - 0.6
        - 0.3
        - 0.1

backend:
  type: in_memory                 -->  InMemoryBackend
  vector_dims:
    vision_0: 2048
    robot_state: 32

write_policy:
  type: on_any_miss               -->  OnAnyMissWritePolicy


6. Three Checkpoint Semantics Summary
====================================================================

Timeline -->

 +---- Cycle N ------------------------------------+    +-- Cycle N+1 --+
 | Stage 1        Stage 2        Stage 3            |    |               |
 | SigLIP+Tok     Gemma 2B       FlowMatch          |    | may be skipped|
 +----+---------------+------------------+----------+    +-------+-------+
      |               |                  |                       |
   [CP1]         [CP2:suspended]      [CP3]                     |
      |                                  |                       |
   On hit:                            On hit:                    |
   Skip S2+S3                         Predict next cycle         |
   Maximum savings                    Schedulable action   <-----+
   Highest risk                       (not fully implemented)

   Applicable: highly               Applicable: consecutive action
   repetitive scenarios              sequences with temporal locality


7. Cache Check Pipeline Detail (Orchestrator.check)
====================================================================

  Orchestrator.check(checkpoint_id, **stage_outputs)
       |
       v
  [1] collect: KeyBuilder.collect(cp_id, **stage_outputs)
       |        Store tensors from stage output into KeyBuilder._cache
       v
  [2] gate: GateFunction(cp_id, cached_data) -> bool
       |        Decide whether a search is worthwhile
       |        (e.g. AlwaysSearchGate always returns True)
       v
  [3] build: KeyBuilder.build(cp_id) -> dict[str, Tensor]
       |        Convert cached_data into fixed-dimension vectors
       |        e.g. vision tokens [1, 256, 2048] -> mean_pool -> [2048]
       |        e.g. robot_state [1, 32] -> squeeze -> [32]
       |
       +--- Gate=SKIP: return MISS (query_keys still built for
       |                            trajectory history recording)
       |
       v (Gate=SEARCH)
  [4] search: SearchStrategy.search(SearchContext)
       |        SearchContext = {query_keys, checkpoint_id,
       |                         current_step, task_key}
       |        Strategy internals:
       |          - Per-field KNN retrieval for each query field
       |          - Weighted RRF fusion across field rankings
       |          - Trajectory-aware: weight last N steps' query_keys
       |        Returns: list[SearchResultLite] (id + score, no payload)
       v
  [5] judge: SimilarityJudge(results, cp_id, cached_data)
       |        ThresholdJudge: top_score >= threshold -> FULL_HIT
       |        Returns: (HitType, winner_id)
       |
       +--- MISS: return CheckResult(MISS, query_keys=keys)
       |
       v (FULL_HIT)
  [6] fetch: CacheStorage.fetch_payload(winner_id)
       |        Retrieve full CachePayload from vector store
       |        (contains action_chunk tensor)
       v
  return CheckResult(FULL_HIT, payload, score, entry_id, query_keys)


8. Core Design Principles
====================================================================

  1. Interceptor Pattern
     InferenceInterceptor implements the BasePolicy interface.
     Fully transparent to WebSocket server and client.
     Zero modification to existing inference pipeline internals.

  2. Two-Phase Retrieval
     search() returns only scores (SearchResultLite).
     fetch_payload() retrieves action tensors (CachePayload) only
     for hit candidates. Avoids unnecessary data transfer.

  3. Episode-Level Batch Write
     During inference, steps are only buffered (StepRecord).
     At episode end, WritePolicy decides whether to batch-write
     to the vector store.
     Entries are built as a linked chain (prev_ids / next_ids).

  4. Trajectory-Aware Search
     SearchStrategy supports trajectory_depth > 1.
     Uses the last N steps' query_keys with weighted fusion.
     trajectory_weights controls weight distribution across steps
     (newest-first).

  5. CP2 Suspended
     PyTorch path's Stage 2 only fills KV cache
     (opaque HuggingFace DynamicCache).
     No usable retrieval key (no autoregressive text generation).
     Awaiting a suitable representation extraction approach.

  6. Fully Pluggable Components
     KeyBuilder / Gate / Judge / SearchStrategy / WritePolicy
     are all selected via YAML config type field.
     Each Checkpoint can be independently configured with
     different component combinations.

  7. Timing System
     Each stage and sub-step has an independent timing probe.
     GPU stages use CUDA Events; CPU stages use perf_counter.
     Per-probe summary is printed automatically at episode end.
