# R16 π0.5 resident: code map for fa_pi_r_f_l_q (read-only)

The model is **PyTorch** (`pi05_libero_pytorch`, eager, SDPA). There is no JAX/XLA compute and no jit on the serving path. One resident process holds **32 arms** (16 libero_10 + 16 libero_spatial).

Each infer request takes one of five routes:
- **cache_blind** and **policy_tail**: CPU only, no GPU work.
- **cache HIT**: stage 1 runs on GPU, then the key is built and the method runs on CPU.
- **policy MISS**: stage 1, stage 2 and stage 3 with K=1 all run on GPU.

Unless prefixed, paths are relative to `S=/home/weiland/projects/openpi_r16_q3_server`. `O=/home/weiland/projects/openpi`.

---

## 1. Launch

**Job spec:** `/nvme/weiland/openpi/eval/fa_pi_r_f_l_q/runs/resident_1_0/server_r16_pi05_k1_1_0.job.json`. `/data/mw_finetune/eval/...` is a symlink alias of `/nvme/weiland/openpi/eval/...`.

**argv:**
```
taskset -c 22-37,66-81 S/.venv/bin/python -m exp.offline_search.rounds.r16.resident
  --registry /data/mw_finetune/eval/fa_pi_r_f_l_q/group.json --server-tag 23150
  --os-method exp.offline_search.rounds.r15.descent.recipe:TaskDescentKnob --os-kwargs '{library:bpool_cs, episode_subset..., task_levels{all ir45}, task_caps[0.1496]*10}'
  --os-cell pi05_l10_cache --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-tokens off
  --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge stall_only
  --os-fit-artifact /home/weiland/projects/openpi_ext/r17_q3_port_1020/fits/filtered/pi05_l10_50_ir45.pkl
  --os-rollout-seed 2026101401 --os-r14-audit --os-log-dir .../runs/resident_1_0 --os-tag r16_pi05_k1_1_0
  --port 23150 --replicas 1 --cache-config /nvme/weiland/openpi/r17_q3_port_1020/definitions/fa_pi_r_f_l_q/fa_pi_r_f_l_q_l50_ir45/config.yaml
  policy:checkpoint --policy.config pi05_libero --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

**env:**
- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`
- `CUDA_VISIBLE_DEVICES=0`
- `OPENPI_SERVER_GPU_MEMORY_LOCK=0`
- `STAGE1_ONLY=0` (full model, no meta stages)
- `GROOT_DENOISING_STEPS=1` (not used by pi05)
- `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` point under `runs/resident_1_0/runtime`
- `HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONPATH` = gr00t paths, S, S/src, S/packages/openpi-client/src

**Where the argv/env come from:**
- `O/exp/offline_search/rounds/r16/conductor_runner.py:47-64` builds the spec: 3 pi05 servers on ports 23150-52, `need_mb=10000`, `OMP=server_threads`.
- `:67-74` (`resident_command`) splices `-m ...r16.resident --registry --server-tag` in place of `serve_pi05.py`.
- `:210-222` starts the servers one at a time through `local_server.start`.
- `O/.../ops/h100/local_server.py:116-132` (command) sets `taskset`, `PYTHONPATH`, `TMPDIR` and the HF offline vars.
- `O/.../ops/h100/node.py:147-201` sets the base argv, `OMP_*`, `GPU_LOCK=0` and `CUDA_VISIBLE_DEVICES=0`.
- `O/.../r17/filter_failed_knob/q3_port_1020/runner.py:11,24` only swaps `SERVER_TREE` to S.

**XLA / JAX:**
- No `XLA_PYTHON_CLIENT_*` variable is set for the server. The only setter in S is `src/openpi/training/data_loader.py:485-486`, which serving does not use.
- JAX is imported (log: "JAX version 0.5.3 available"), but the serving path only calls `jax.tree.map` (`policy.py:79`, `interceptor.py:1688`, `stage_io.py:72`).
- JAX inference is hard-disabled at `src/openpi/policies/policy_config.py:56-59`.
- I could not confirm at runtime whether an XLA GPU backend is ever initialised. Indirect evidence says no 75% preallocation happens: `need_mb=10000` per resident, and the conductor comment "pi05 R3: 23 of 24 GB" for 3 residents (`conductor_runner.py:192`).
- The GPU today is an RTX 4090 with 49140 MiB (nvidia-smi). Only GR00T residents were running when I checked, so I could not measure pi05 memory.

**Entry chain:**
1. `exp/offline_search/rounds/r16/resident.py:148-163` (main): reads the group, monkeypatches `plugin.install` (`:157`), then calls `serve_pi05.main`.
2. `exp/offline_search/closed_loop/serve_pi05.py:26-55`:
   - `stage_overrides.parse_flags`: stage1 mode is "full" by default.
   - `plugin.parse_cli`, then tyro parses `scripts/serve_policy.Args`.
   - `cost.install_startup_hook` (`stage_overrides.py:52-61`) only adds fields to the startup log row.
   - `cost.install_pi05("full")` is a no-op (`stage_overrides.py:306-307`).
   - `plugin.install(...)` resolves to `resident.install` (`:53`).
   - `sp.main(args)` (`:55`).
3. `scripts/serve_policy.py:840-845` (main) calls `_serve_single` (`:947`):
   - `_configure_server_gpu_memory_lock` (`:334-381`): off here.
   - Monitor level is OFF (`:817`).
   - `create_policy` (`:269-317`) calls `create_trained_policy(_disable_compile_for_serving(get_config("pi05_libero")), dir)`. `_disable_compile_for_serving` (`:213-225`) forces `pytorch_compile_mode=None`.
   - Shared storage is built from the startup yaml (`:1027-1033`).
   - `BatchingCoordinator` (`:1044-1062`).
   - The per-connection factory is `_wrap_policy(eager=True)` (`:1064-1069`).
   - `WebsocketPolicyServer(concurrent=True)` (`:1074-1082`), then `serve_forever` (`:1100`).

**Model load:**
- `policy_config.py:53-55`: `train_config.model.load_pytorch` → `models/model.py:243-247` (`PI0Pytorch(config)` + `safetensors.load_model`), then `to_bfloat16_for_selected_params("bfloat16")`.
- The device falls to "cuda" (`policy_config.py:72-78`); `Policy` calls `.to(device).eval()` (`policy.py:58-60`).
- Norm stats fall back to `S/assets/pi05_libero/physical-intelligence/libero` (log lines; fallback at `policy_config.py:66-69`).
- Numerics-relevant settings:
  - SDPA is fixed (`pi0_pytorch.py:224-225`).
  - `torch.set_float32_matmul_precision("high")` (`:238`).
  - The `transformers_replace` patch is required (`:245-252`).
  - Config: `action_horizon=10`, `max_token_len=200`, `pi05=True` (`training/config.py:795-796`, log).

**Loading all arms in one resident:** `resident.install` (`resident.py:79-145`)
- Loops over all 32 `group['entries']` (`suite=None` for pi05). For each entry:
  - checks the fit/yaml sha256 (`:93-97`);
  - builds the arm's argv (`:98-101`) and runs `plugin.parse_cli`;
  - sets `opts.os_miss_steps=k` (`:105`);
  - creates `plugin.PluginRuntime(opts,"pi05")` (`:110`).
- `PluginRuntime` (`plugin.py:494-655`):
  - loads the fit pickle (`_load_and_fit`, `:658-727`) and the store libraries;
  - calls `seed_policy_process(2026101401)` (`:639-641`; `rollout_seed.py:7-19`). This runs 32 times, before the model load;
  - logs "osplug ready" (`:654`).
- Arrays are deduplicated (`ArrayPool`, `resident.py:20-61,113-115`).
- `plugin.RUNTIME = Registry(runtimes)`.
- `build_per_connection_components` is patched to `registry.attach(..., yaml_id)` (`:125-130`).
- `WebsocketPolicyServer` is replaced by `ResidentServer`, which wraps the factory with `plugin._wrap_factory` (`:131-139`).
- Writes `resident/23150.json` (`:141-144`).

---

## 2. Wire protocol and session state

**Encoding:** `openpi_client.msgpack_numpy` (`S/packages/openpi-client/src/openpi_client/msgpack_numpy.py:21-57`). An ndarray is sent as `{__ndarray__, data(bytes), dtype(str), shape}`; numpy scalars as `__npgeneric__`.

**Server socket** (`src/openpi/serving/websocket_policy_server.py`):
- `compression=None`, `max_size=256MB`, `/healthz` (`:534-545`, `:1093-1098`).
- On connect the server sends the metadata dict first (`:637`).
- Request loop: `:657-1020`.

**Message types:**

| `__ctrl__` / kind | Handler | Behaviour | Reply |
|---|---|---|---|
| `episode_start` | `:664-694` | Binds the bundle lazily if none is bound yet (uses `bundle_id` from the message, else `"default"`), then calls `on_episode_start(experiment, task, episode_id, episode_name, extra_metadata=__extra__)` | `{"__ack__":"episode_start"}` |
| `episode_end` | `:695-709` | `on_episode_end(success=__success__)` | ack |
| `prefill_trajectory` | `:710-746` | Prefill | ack |
| `load_cache_config` | `:747-857` | `{yaml_path \| yaml_content, yaml_id, bundle_id}`: `load_cache_config` → `build_shared_storage` (in a thread) → registers in `_bundles[bundle_id]` (`:827-843`) | `{"__ack__":"load_cache_config", version, bundle_id}` |
| `select_bundle` | `:858-911` | `_bind_bundle(bundle_id)` (`:600-626`): factory → `on_task_end` on the old stack → `on_task_begin` on the new one | ack |
| `fetch_dump`, `preload_normalizer_buffer`, `unload_warmup_buffer`, `dump_metrics`, `snapshot_mem`, `mem_history_start/stop`, `set_batch_params`, `throughput_summary` | `:912-978` | Admin / monitoring | ack |
| any other ctrl | `:979-980` | — | `{"__ack__":"ignored"}` |
| non-ctrl dict = **infer** | `:983-1019` | Default-binds if unbound, then `await asyncio.to_thread(conn_policy.infer, obs)` (`:1006`) | Adds `server_timing{infer_ms, prev_total_ms}` (`:1012-1017`) |

- On an exception the server sends a traceback string frame and closes with code 1011 (`:1055-1062`).
- Client calls: `websocket_client_policy.py:46-54` (infer), `:56-92` (episode start/end), `:160-206` (`load_cache_config`, `select_bundle`).

**Actual client sequence:**
1. The driver preloads all 32 YAMLs over one control connection: `ctl.load_cache_config(row.yaml, yaml_id=arm, bundle_id=arm)` (`O/exp/offline_search/rounds/r16/conductor.py:415-423`, called at `:478`). The log shows bundles v1..v32.
2. Each worker (`O/examples/libero/episode_runner.py`):
   - calls `select_bundle(task.bundle_id)` only when the bundle changes (`:191-198`);
   - calls `episode_start(experiment, task_description, episode_idx, extra={task_id, orig_init_state_idx, task_uid, attempt})` (`:114-139`, `:217-222`);
   - sends `infer(element)` every 5 steps (`O/examples/libero/main.py:315-331`). `element = {observation/image uint8 224×224×3, observation/wrist_image, observation/state (8,), prompt str}`, with no `__extra__`, so `decision_id` is None and `executed_steps` defaults to 5;
   - calls `episode_end(success)` (`episode_runner.py:313`).

**Infer reply:**
- `{state, actions (10×7 after LiberoOutputs), __hit_meta__{hit_type, winner_id, cp1_score, checkpoint, score, searched, factor_outputs{osplug: wire_diag}, [miss_nfe]}, server_timing}`.
- `__hit_meta__` is built at `interceptor.py:916-1000` and `plugin.py:2100-2102`; `wire_diag` at `plugin.py:1251-1259`.

**State the server keeps:**
- **Per process:**
  - base `Policy` (model and transforms);
  - one `BatchingCoordinator` (`serve_policy.py:1049`);
  - module globals `_bundles` / `_current_bundle` (`websocket_policy_server.py:89-99`);
  - `BackendPool` singleton, one backend per pkl fingerprint (`cache/backend_pool.py`; log loads the libero_10 pkl with 2640 entries and the libero_spatial pkl with 1018);
  - `plugin.RUNTIME = Registry` of 32 `PluginRuntime`s. Each holds the fitted method, tables, decision-log writer, `decision_count` and a sessions WeakSet (`plugin.py:494-655`).
- **Per connection**, created in `_bind_bundle` → `_wrap_factory` (`plugin.py:2350-2361`) → `_wrap_policy` (`serve_policy.py:549-788`). Stack from outside in:
  - `_ConnPolicy` (`plugin.py:2117-2347`);
  - `WarmResetEvidencePolicy` (from the yaml `miss` block; `cache/warm_reset/runtime.py:283-317`, `evidence.py:78-255`);
  - `InferenceInterceptor` (`serve_policy.py:657-675`), which owns a `SystemTimer` and stage functions bound to the coordinator;
  - `CacheOrchestrator` (`serve_policy.py:623-633`) with per-connection components (`cache/config.py:4384+`): storage facade, `CP1SpatialPool16KeyBuilder`, `always_search` gate, and the plugin replacements `PluginStrategy` / `PluginJudge` / `PluginStorage` (`plugin.py:796-822`);
  - `PluginSession` (`plugin.py:869-911`): method deepcopy with shared arrays (`clone_method`, `:437-447`), RLock;
  - `_BlindAdapter` (`plugin.py:1829-1942`), only when `rt.r4`.
  - Handler locals: `conn_policy`, `bound_bundle_id`, `prev_total_time`.
- **Per episode:**
  - The session resets lazily at the first decision after `episode_start`: `client_episode_start` (`:913-917`) then `_begin` (`:925-977`) creates `EpisodeView`, the `_Buf` histories (`b_v0`, `b_v1`, `b_rs`, `b_raw`, `b_aex`), `hits`, `has_vision`, the policy tail, and calls `method.reset`. That reset is `Q3StallInner.reset`, which builds a new `StallTracker` (`stall_only.py:60-64`).
  - Orchestrator buffers reset (`orchestrator.py:494-519`, `646-662`).
  - Evidence episode (`evidence.py:197-222`).

---

## 3. Infer path end to end

**Call chain:** `ws._handler` `:1006` → `_ConnPolicy.infer` (`plugin.py:2182-2186`, under the per-connection lock) → `_osp_infer` (`:2188-2344`):
1. `set_obs(obs)` (`:1261`).
2. Reserves `decision_index` (`:2237-2239`).
3. If `rt.blind`: `_try_blind` (`:1965-2105`).
4. Else, or if that returned None: `WarmResetEvidencePolicy.infer` → `InferenceInterceptor.infer` (`interceptor.py:1645-2188`).
5. `after_infer` emits the decision row (`:1274-1450`).

**Blind route (CPU only, no model call; `_try_blind`):**
- `adapter.prepare` runs the full input transform chain just to get the normalised state (`plugin.py:1908-1917`).
- The look reason is chosen (`:1997-2027`): step 0, previous MISS without a tail, or a judge requirement forces a LOOK.
- Otherwise it calls `method.blind_step` or `policy_tail_step`:
  - `TaskDescentKnob` → `R10Recipe` (`r10/recipe/recipe.py:135-142`) → `Q3StallInner` (`r06/ideation_Q1/method_c/stall_only.py:96-105`) → `BlindAWM.blind_step` (`r04/k1_blind/blind_awm.py:164-231`).
  - The policy tail re-serves rows 5..9 of the last MISS wire action (`closed_loop/blind.py:47-60`, `plugin.py:2058-2061`).
- Then `adapter.output` runs the output transform (`:1931-1938`), `commit_external_hit` (`:1810-1826`) and `broadcast_action` / `clear`.
- The evidence wrapper is bypassed on this route.

**Vision route** (`interceptor.py:1688-1871`):
1. Input transforms on CPU (`:1688-1689`). The chain is `InjectDefaultPrompt` → `LiberoInputs` (third image is all zeros and masked) → `Normalize` (quantile for pi05, `training/config.py:189`) → `ResizeImages(224)` → `TokenizePrompt(200)` → `PadStatesAndActions(32)` (`training/config.py:128-139`). Leaves are unbatched CPU tensors (`:1716-1722`).
2. **Stage 1 on GPU**, through the coordinator (`:1737`, `:434-435`). `_BlindAdapter` wraps this call with a counter and a **device-wide `torch.cuda.synchronize()`** that sets `s1_ms` (`plugin.py:1866-1906`).
3. **CP1 check** (`:1741-1753`) → `orchestrator._check_impl` (`orchestrator.py:979-1328`):
   - **Key construction:** `key_builder.collect(stage1)` then `build()` (`:1009-1025`). It runs on the request thread from **stage-1 output `prefix_embs[0]`** (`components/key_builder.py:180-205`, `279-316`):
     - `vision_0` = tokens [0:256] and `vision_1` = tokens [256:512], each pooled with `adaptive_avg_pool2d` 16×16 → 4×4, giving a 32768-d vector (`:217-238`, `444-456`);
     - `robot_state` = `stage1.state[0]` (normalised, padded to 32);
     - pooling runs on GPU, then one D2H `.cpu().float()` (`:241-243`);
     - enabled fields are `vision_0`, `vision_1`, `robot_state` (`prompt_emb` and `vision_2` are disabled in the yaml).
   - **Gate:** `always_search`.
   - **Search:** `PluginStrategy.search` → `PluginSession.on_search` (`plugin.py:1076-1198`):
     - `_push_inputs` appends the keys to the episode buffers (`:1050-1074`);
     - `method.query(OnlineQueryView)` runs entirely on CPU in numpy (`q_us` ≈ 4 ms in the log).
   - **What `method.query` does:**
     - `R14Knob.query` (`r14/knob/recipe.py:92-123`) → `R10Recipe.query` (`r10/recipe/recipe.py:125-128`) → `Q3StallInner.query` (`stall_only.py:66-94`) → `_CorrectedCache.query` (`r10/recipe/recipe.py:65-73`) → `AWM.query/_dist` (`r02/g1_awm/awm.py:442-513`). `_dist` projects the keys with `B0T`/`B1T` (64×32768 float32) and computes kernel-weighted distances over the fitted task tables.
     - **Corrector** (`_CorrectedCache._correction`, `r10/recipe/recipe.py:38-58`): a random-feature regression head per task, gated by donor distance, `blend=0.5`. No model call.
     - **Q3 stall gate:** `StallTracker.observe(q, step*5)` (`r06/ideation_Q3/stall/stall.py:475-506`). On `slow_confirmed` it sets `os_force_miss=1`, `os_reason=63` (`stall_only.py:79-91`).
     - **Knob layer:** `propose()` (`r14/knob/controller.py:15-29`) may set `os_force_miss=1`, `os_reason=73` (`r14/knob/recipe.py:110-113`).
     - **Diagnostics:** `R14Knob._diagnostics` always runs `original_corrector_audit` (`:125-192`), a second CPU corrector evaluation for logging only.
   - **Library search:** the "library" searched is the method's own fitted tables from `--os-root` (`bpool_cs` rows). The in-memory backend (`/nvme/weiland/openpi/r17_library/libero/pi05/libero_10/cp1_spatial_pool_16.pkl`, symlinked from the yaml `preload_path`) is **not** searched:
     - the native shadow search is off (`--os-no-shadow-native`, `plugin.py:1138-1144`);
     - the backend is only validated once per backend (`plugin.py:772-794`);
     - the winner is a synthetic payload, because `lib=bpool_cs` and `act` is not None (`plugin.py:1120-1128`, `1214-1221`).
   - **Judge:** `_verdict` (`plugin.py:987-1038`): for `stall_only`, a forced flag means MISS, otherwise HIT (`:1028-1029`). `PluginJudge` maps that to `FULL_HIT` or `MISS` (`:1776-1787`) and never returns `WARM_START`.
4. **HIT:** `fetch_payload` → `synthetic_payload` → `broadcast_action` → `PluginStrategy.record_action` → `session.on_executed` (`plugin.py:1223-1249`) → `buffer_for_write` → output transform → return. Stages 2/3 are skipped (`interceptor.py:1805-1871`).
5. **MISS (GPU):**
   - Stage 2 through the coordinator (`:1916-1917`).
   - The request thread samples noise with `model.sample_noise` (global CUDA RNG, `torch.normal`; `interceptor.py:2048-2058`, `pi0_pytorch.py:316-333`).
   - Stage 3 through the coordinator with `num_steps=self._miss_steps()` (`:2059-2064`).
   - CP3 is not configured, so the check is a no-op (`orchestrator.py:1000-1003`).
   - `action_chunk[0].cpu()`, the intermediates (only `{0.5: noise}` at K=1) and `broadcast_action` follow (`:2115-2144`).
   - Output transform (`:2160-2161`); `miss_nfe` is added to `__hit_meta__` (`:2176-2178`).

**Where K is applied:**
1. yaml `miss.num_steps: 1` → `MissSpec` → `miss_num_steps` (`warm_reset/runtime.py:290-317`).
2. → `InferenceInterceptor(miss_num_steps=1)` (`serve_policy.py:674`; `interceptor.py:342-355`, `1075-1077`).
3. → `Stage3MissPayload.num_steps` (`interceptor.py:460-472`).
4. → bucket key `("miss",None,1)` (`batching_coordinator.py:100-101`).
5. → `model.run_stage3(num_steps=1)` (`:158-163`) → `range(num_steps)` loop (`pi0_pytorch.py:806-813`).
- `resident.py:105` (`os_miss_steps`) and `stage_overrides.miss_steps_from_yaml` only affect logging.

---

## 4. Batching

**Setup:**
- One `BatchingCoordinator` per resident process, shared by **all connections and all arms** (`serve_policy.py:1044-1061`, passed via `shared_cache["coordinator"]` to every interceptor at `:587/664`). `bundle_id` is bookkeeping only (`batching_core.py:192-212`).
- `max_batch_size=8`, `max_wait_ms=10`, from env `BATCHING_MAX_BATCH_SIZE` / `BATCHING_MAX_WAIT_MS` defaults (`serve_policy.py:1047-1048`; log confirms).
- One worker thread per stage (`BATCHING_STAGE{1,2,3}_WORKERS=1`). Each worker has its own CUDA stream (`batching_core.py:321-347`; log: "per-worker CUDA streams enabled").
- Default scheduler is pull-then-group (`:884-898`): block for the first request, then collect until the batch has 8 or 10 ms have passed since that request. The bucket-first stage-3 scheduler is opt-in and off (`:881`).

**Per batch:** run on the worker stream (`:915`), `_sync_stage_stream` (`:1042-1054`), then publish replies (`:1033-1040`). The loop runs under `torch.no_grad()` (`:751`).

**What each stage computes** (`batching_coordinator.py:117-208`, `pi0_pytorch.py`):

| Stage | Model call | Computation | Stacking / splitting |
|---|---|---|---|
| 1 | `run_stage1` (`:592-606`) | `preprocess_observation_pytorch` + SigLIP `embed_image` for all 3 cameras (the dummy third camera is still encoded, but masked) + language embedding × √d + masks / position ids (`:348-397`, `537-544`). Output `prefix_embs` is [B, 968, 2048] bf16. | `stack_observation`: `torch.stack` + `.to(cuda)` (`stage_io.py:45-72`); split into [1,…] **views** (`:99-110`) |
| 2 | `run_stage2` (`:608-620`) | PaliGemma (Gemma-2B) prefix forward with `use_cache` → `DynamicCache` (`:546-560`) | `torch.cat` stage-1 shards; `DynamicCache.batch_split` + **clone** (`stage_io.py:80-96`, `149-165`, `196-221`) |
| 3 | MISS: `run_stage3(return_intermediates=True)` (`pi0_pytorch.py:644-702`, `771-815`) | `denoise_step` of the action expert (Gemma-300M with adaRMS) for K steps | noise `torch.stack` (`batching_coordinator.py:153-168`); grouped by `bucket_key` (`batching_core.py:1134-1148`) |

**WARM_START path:** `Stage3WarmStartPayload` (`batching_core.py:113-128`) with key `("warm_start", start_t, num_steps)` → `run_stage3_from` (`batching_coordinator.py:188-208`; `pi0_pytorch.py:704-769`). It is unused here, since `PluginJudge` never returns WARM_START. Warm-reset payloads (`batching_core.py:130-143`) are also unused.

**Unconfirmed observations:**
- The MISS payload is built without `ready_events` (`interceptor.py:460-471`), while noise is sampled on the request thread's stream.
- `asyncio.to_thread` uses the default executor, which caps concurrency at about 32 threads (Python behaviour).
- Batch width can change bf16 GEMM results (comment at `stage_overrides.py:280-281`).

---

## 5. Model staging, jit and shapes

- Stage split: `run_stage1`, `run_stage2`/`run_stage2_capture`, `run_stage3`/`run_stage3_from` (`pi0_pytorch.py:592-769`). Data types: `Stage1Output`, `Stage2Output`, `Stage3Output` (`:21-109`).
- No JAX jit. The JAX staged jits in `policy.py:63-74` are dead code for PyTorch.
- No `torch.compile`: compile mode is None (`serve_policy.py:213-225`), the interceptor runs eager (`interceptor.py:397-401`) and is then rebound to the coordinator (`:474-476`). `cudagraph_mark_step_begin` is called (`:1733`) but there are no cudagraphs.
- No padding or bucketing on the batch dimension: B is 1..8 as collected.
- Fixed per-request shapes: images 224², 200 tokens, prefix 768+200 = 968, state 32, actions [10, 32].

---

## 6. Arm and knob selection

- Arm = `yaml_id` of the bound bundle. `select_bundle(arm)` → `get_current_cache_bundle(bundle_id)` (`serve_policy.py:600`) → patched `build_per_connection_components` → `Registry.attach(yaml_id)` (`resident.py:69-72`) → `runtimes[arm].attach` builds a `PluginSession` with that arm's method clone.
- An unknown `yaml_id` raises. If no bundle was selected, the code falls back to the **latest** loaded bundle (`serve_policy.py:600`).
- The arm is fixed per connection until the next `select_bundle`, which rebuilds the stack.
- One resident **does** serve different arms (IR levels, l10/spatial, library sizes) on different connections at the same time, and their GPU requests batch together. Each arm has its own `decision_count`, decision log and fitted method. Per-arm plugin flags can differ: in `fa_cs_pi_rg` the cg/crg arms run without `--os-blind` (`O/.../q3_port_1020/build.py:291-296`).
- Method naming:
  - `Q3Knob_irXX__...` comes from `build.py:87-105`.
  - `Q3CS_cache__` / `Q3CS_corrected__` (R10Recipe + `Q3StallInner`, no knob) come from `:108-119`.
  - Confirmed by unpickling: `TaskDescentKnob(r15)` → `Q3StallInner` → `_CorrectedCache(BlindAWM, AWM)`, `B0T` shape (64, 32768).

---

## 7. Per-request timing today

- **Decision logs:** `<group>/runs/<arm>/server_<port>/decisions_<arm>_<port>.jsonl`, emitted at `plugin.py:730-746` and `1338-1429`. Fields:
  - `infer_ms`: plugin `_osp_infer` wall time, t0 at `:2220`, dt at `:2317`;
  - `pre_ms`, `post_ms`, `s1_ms` (stage 1 + device sync), `s23_ms`, `q_us`, `search_us`, `blind_prepare_ms`, `blind_output_ms`;
  - `ts` from `time.time()`;
  - `src` ∈ {cache, cache_blind, policy_tail, policy}, `judge` force:63 (Q3) / force:73 (knob).
  - Observed counts for l50_ir45: policy_tail 4026, policy/73 3670, cache 554, cache_blind 542, policy/63 412.
- **Evidence:** a `wall_ms` row per non-blind decision, written to the yaml `miss.evidence_dir` (`evidence.py:150-188`).
- **Wire:** `server_timing.infer_ms` (`websocket_policy_server.py:1012`).
- **Not recorded:**
  - `SystemTimer`: disabled because the yaml sets `timer.enabled:false`.
  - Monitor: OFF, so there are no batch records and no `[batch_done]` log lines (`batching_core.py:371-376`, `903`).
  - `serving.jsonl` / `batching.jsonl` do not exist in this tree. They exist only in `O/exp/r17_integration/models/fastwam/formal_batch_server.py:122,158` and an r17 pi05 test.
- **Client side:** `_ct_infer` (`main.py:325-336`) → `client_timing` row (`episode_runner.py:337-341`).
- **Best arrival → sent hooks:**
  - Right after `await websocket.recv()` (`websocket_policy_server.py:660`). Note that the current `start_time` at `:659` is taken *before* recv, so it includes idle time.
  - Right after `await websocket.send(packer.pack(action))` (`:1019`).

---

## Minimal module set for identical per-request computation

- `openpi.policies.policy_config.create_trained_policy` + `openpi.training.config.get_config("pi05_libero")` with compile disabled (`serve_policy._disable_compile_for_serving`). This gives the transforms and `PI0Pytorch`. Use S's `.venv`, because the transformers_replace patch is required.
- `openpi.models_pytorch.pi0_pytorch` (`run_stage1/2/3`, `sample_noise`) + `openpi.serving.stage_io` (or `batching_coordinator.Pi05StageBatcher`) for identical stack/split.
- `openpi.cache.config` (`load_cache_config`, `build_shared_storage`, `build_per_connection_components`), `openpi.cache.orchestrator.CacheOrchestrator`, `openpi.cache.interceptor.InferenceInterceptor`, `openpi.cache.warm_reset.pi05.build_pi05_warm_reset` (for `miss_num_steps`).
- `exp.offline_search.closed_loop.plugin` (`PluginRuntime`, `PluginSession`, `PluginStrategy`/`Judge`/`Storage`, `_ConnPolicy`, `_BlindAdapter`, `_try_blind`, `_wrap_factory`) + `exp.offline_search.rounds.r16.resident.install` (multi-arm Registry).
- The method classes come in automatically when the fit pickles are loaded: r15/r14/r13/r11/r10 recipes, `r04 blind_awm`, `r02 awm`, `r06 stall_only` / `stall_bridge` / `stall`, `r05 q1_commit.judge`, `harness.api` / `dims` / `store`, `closed_loop.blind`, `closed_loop.rollout_seed`.
- `openpi_client.msgpack_numpy` for the wire format.

The lowest-risk route is to reuse `resident.install` and `_wrap_policy` as they are, and replace only `WebsocketPolicyServer` and `BatchingCoordinator`. Two things block bit-exact replay:
- **Noise:** it comes from the process-global CUDA RNG in global MISS-arrival order. The interceptor accepts a `noise=` argument (`interceptor.py:1645`, `1726-1730`), but the server never passes one.
- **Batch composition:** it may change bf16 results (unconfirmed by measurement).
