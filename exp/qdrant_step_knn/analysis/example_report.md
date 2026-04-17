# 实验报告

## 配置

```json
{
  "candidate_limit": 50,
  "db_data_dir": "data/libero_spatial",
  "mode": "named",
  "multivector_collection": "openpi_steps_multivector",
  "named_collection": "openpi_steps_named",
  "query_file": "data/libero_spatial_1/episode_0000_20260331_035016_586551.h5",
  "rrf_k": 60,
  "selected_keys": [
    "vision_0",
    "robot_state"
  ],
  "source_config": {
    "cache": {
      "dir": ".cache/qdrant_step_knn",
      "enabled": true,
      "refresh": false
    },
    "db_data_dir": "data/libero_spatial",
    "experiment": {
      "candidate_limit": 50,
      "max_query_steps": null,
      "mode": "named",
      "query_step_idxs": [],
      "rrf_k": 60,
      "step_filter": "all",
      "step_window": 1,
      "top_k": 1
    },
    "keys": {
      "clean_action": {
        "enabled": false,
        "weight": 2.0
      },
      "noise_action_1": false,
      "noise_action_2": false,
      "noise_action_3": false,
      "noise_action_4": false,
      "noise_action_5": false,
      "noise_action_6": false,
      "noise_action_7": false,
      "noise_action_8": false,
      "noise_action_9": false,
      "prompt_emb": {
        "enabled": false,
        "weight": 1
      },
      "robot_state": {
        "enabled": true,
        "weight": 10
      },
      "vision_0": {
        "enabled": true,
        "weight": 1
      },
      "vision_1": {
        "enabled": false,
        "weight": 2.0
      },
      "vision_2": {
        "enabled": false,
        "weight": 1.0
      }
    },
    "output_dir": "exp/qdrant_step_knn/data/example",
    "qdrant": {
      "grpc_port": 6334,
      "multivector_collection": "openpi_steps_multivector",
      "named_collection": "openpi_steps_named",
      "prefer_http": false,
      "request_timeout": 1800,
      "url": "http://155.98.36.47:6333"
    },
    "query_file": "data/libero_spatial_1/episode_0000_20260331_035016_586551.h5"
  },
  "step_filter": "all",
  "step_window": 1,
  "top_k": 1,
  "transport": "grpc",
  "weights": {
    "robot_state": 0.9090909090909091,
    "vision_0": 0.09090909090909091
  }
}
```

## 每个 Step 结果

| query_step | named_topk_clean_action_l2_mean | multivector_topk_clean_action_l2_mean |
| --- | ---: | ---: |
| step_0000 | 0.446796 | - |
| step_0001 | 0.509857 | - |
| step_0002 | 1.015777 | - |
| step_0003 | 1.181001 | - |
| step_0004 | 1.512508 | - |
| step_0005 | 0.809676 | - |
| step_0006 | 0.763245 | - |
| step_0007 | 3.278479 | - |
| step_0008 | 2.987662 | - |
| step_0009 | 1.237496 | - |
| step_0010 | 1.049473 | - |
| step_0011 | 0.658738 | - |
| step_0012 | 1.230878 | - |
| step_0013 | 0.922679 | - |
| step_0014 | 2.385841 | - |
| step_0015 | 5.369176 | - |

## 平均结果

| 指标 | 数值 |
| --- | ---: |
| named 平均 top-k clean action L2 | 1.584955 |
| multivector 平均 top-k clean action L2 | - |

说明：L2 越小越好。
