# docs/cache/

Cache system user guides: tutorial, migration guide, per-component docs.

| File | Description |
|------|-------------|
| [tutorial.md](tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing |
| [migration.md](migration.md) \[[EN](migration.en.md)\] | Cache framework migration guide: how to adapt the cache system for non-Pi0.5 models |
| [temporal_prune.md](temporal_prune.md) \[[EN](temporal_prune.en.md)\] | Temporal Prune KeyBuilder 使用指南：两步架构、参数配置、Reducer 选择、离线 Artifact 构建、生命周期 |
| [llm_layer_extract.md](llm_layer_extract.md) \[[EN](llm_layer_extract.en.md)\] | CP1 LLM Layer Extract KeyBuilder 使用指南：两步架构（LayerExtractor + PrefixReducer）、attach_model 注入、离线 Stage 1 重建契约（重 tokenize + tokenizer self-check）、在线/离线 parity test |
| [verdict_factor_judge.md](verdict_factor_judge.md) \[[EN](verdict_factor_judge.en.md)\] | **2026-05-07 重构 (G1 APPROVED Round 4)**：5 → 17 因子扁平化 (`<descriptor>_<source>_<channel>` + `topk_action_variance`)；judge 4 层正交架构 (Normalization → Factor → Calibration → Composer)；no cold-start (启动 fail-fast，废 `cold_start_strategy` / `all_nan_fallback`)；wire schema_version=2 (`factor_outputs.{raw, calibrated, composer_score}`)；end-to-end build pkl + enrich-existing-pkl + warmup → eval handshake + 自定义扩展指南 |

See also: [../architecture/cache_system.md](../architecture/cache_system.md) for the design spec.

Back to [docs index](../README.md).
