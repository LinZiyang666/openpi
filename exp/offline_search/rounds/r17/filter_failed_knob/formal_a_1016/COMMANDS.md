# Queue-ready commands — coordinator review

Run from `/home/weiland/projects/openpi`. The parent `filter_failed_knob/commands` aliases match chain_ffk.run.sh. Only the coordinator executes prepare/launch.

## ffk1015_dp_libero_reduced_formal_A_original_levels

RTX 4090; R4/B8; workers {'local': 24, 'timan1': 24}; episodes 5000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_libero_reduced_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_libero_reduced_formal_A_original_levels.sh
```

## ffk1015_dp_libero_reduced_formal_A_filtered_levels

RTX 4090; R4/B8; workers {'local': 24, 'timan1': 24}; episodes 5000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_libero_reduced_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_libero_reduced_formal_A_filtered_levels.sh
```

## ffk1015_dp_libero_full_formal_A_original_off

RTX 4090; R4/B8; workers {'local': 24, 'timan1': 24}; episodes 6000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_libero_full_formal_A_original_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_libero_full_formal_A_original_off.sh
```

## ffk1015_dp_libero_full_formal_A_original_levels

RTX 4090; R4/B8; workers {'local': 24, 'timan1': 24}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_libero_full_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_libero_full_formal_A_original_levels.sh
```

## ffk1015_dp_libero_full_formal_A_filtered_off

RTX 4090; R4/B8; workers {'local': 24, 'timan1': 24}; episodes 6000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_libero_full_formal_A_filtered_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_libero_full_formal_A_filtered_off.sh
```

## ffk1015_dp_libero_full_formal_A_filtered_levels

RTX 4090; R4/B8; workers {'local': 24, 'timan1': 24}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_libero_full_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_libero_full_formal_A_filtered_levels.sh
```

## ffk1015_fastwam_libero_reduced_formal_A_original_levels

RTX 4090; R2/B16; workers {'timan1': 30, 'timan107': 20, 'timan108': 20}; episodes 30000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_libero_reduced_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_libero_reduced_formal_A_original_levels.sh
```

## ffk1015_fastwam_libero_reduced_formal_A_filtered_levels

RTX 4090; R2/B16; workers {'timan1': 30, 'timan107': 20, 'timan108': 20}; episodes 22500; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_libero_reduced_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_libero_reduced_formal_A_filtered_levels.sh
```

## ffk1015_fastwam_libero_full_formal_A_original_off

H100; R2/B16; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes 6000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_libero_full_formal_A_original_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_libero_full_formal_A_original_off.sh
```

## ffk1015_fastwam_libero_full_formal_A_original_levels

H100; R2/B16; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_libero_full_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_libero_full_formal_A_original_levels.sh
```

## ffk1015_fastwam_libero_full_formal_A_filtered_off

H100; R2/B16; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes 6000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_libero_full_formal_A_filtered_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_libero_full_formal_A_filtered_off.sh
```

## ffk1015_fastwam_libero_full_formal_A_filtered_levels

H100; R2/B16; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_libero_full_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_libero_full_formal_A_filtered_levels.sh
```

## ffk1015_cosmos_libero_reduced_formal_A_original_levels

H100; R8/B1; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_cosmos_libero_reduced_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_cosmos_libero_reduced_formal_A_original_levels.sh
```

## ffk1015_cosmos_libero_reduced_formal_A_filtered_levels

H100; R8/B1; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_cosmos_libero_reduced_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_cosmos_libero_reduced_formal_A_filtered_levels.sh
```

## ffk1015_cosmos_libero_full_formal_A_original_off

H100; R8/B1; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes 6000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_cosmos_libero_full_formal_A_original_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_cosmos_libero_full_formal_A_original_off.sh
```

## ffk1015_cosmos_libero_full_formal_A_original_levels

H100; R8/B1; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_cosmos_libero_full_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_cosmos_libero_full_formal_A_original_levels.sh
```

## ffk1015_cosmos_libero_full_formal_A_filtered_off

H100; R8/B1; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes 6000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_cosmos_libero_full_formal_A_filtered_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_cosmos_libero_full_formal_A_filtered_off.sh
```

## ffk1015_cosmos_libero_full_formal_A_filtered_levels

H100; R8/B1; workers {'timan1': 35, 'timan107': 25, 'timan108': 25}; episodes <= 30000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_cosmos_libero_full_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_cosmos_libero_full_formal_A_filtered_levels.sh
```

## ffk1015_pi05_libero_reduced_formal_A_original_levels

RTX 4090; R3/B1; workers {'timan107': 64}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_reduced_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_reduced_formal_A_original_levels.sh
```

## ffk1015_pi05_libero_full_formal_A_original_off

H100; R3/B1; workers {'timan108': 39}; episodes 4000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_full_formal_A_original_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_full_formal_A_original_off.sh
```

## ffk1015_pi05_libero_full_formal_A_original_levels

H100; R3/B1; workers {'timan108': 39}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_full_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_full_formal_A_original_levels.sh
```

## ffk1015_pi05_libero_full_formal_A_filtered_off

H100; R3/B1; workers {'timan108': 39}; episodes 4000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_full_formal_A_filtered_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_full_formal_A_filtered_off.sh
```

## ffk1015_pi05_libero_full_formal_A_filtered_levels

H100; R3/B1; workers {'timan108': 39}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_full_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_full_formal_A_filtered_levels.sh
```

## ffk1015_groot_libero_reduced_formal_A_original_levels

RTX 4090; R4/B1; workers {'timan107': 64}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_reduced_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_reduced_formal_A_original_levels.sh
```

## ffk1015_groot_libero_reduced_formal_A_filtered_levels

RTX 4090; R4/B1; workers {'timan107': 64}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_reduced_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_reduced_formal_A_filtered_levels.sh
```

## ffk1015_groot_libero_full_formal_A_original_off

RTX 4090; R4/B1; workers {'timan107': 64}; episodes 4000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_full_formal_A_original_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_full_formal_A_original_off.sh
```

## ffk1015_groot_libero_full_formal_A_original_levels

RTX 4090; R4/B1; workers {'timan107': 64}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_full_formal_A_original_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_full_formal_A_original_levels.sh
```

## ffk1015_groot_libero_full_formal_A_filtered_off

RTX 4090; R4/B1; workers {'timan107': 64}; episodes 4000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_full_formal_A_filtered_off.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_full_formal_A_filtered_off.sh
```

## ffk1015_groot_libero_full_formal_A_filtered_levels

RTX 4090; R4/B1; workers {'timan107': 64}; episodes <= 16000 after off; WAITING_OFF.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_full_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_full_formal_A_filtered_levels.sh
```

## ffk1015_pi05_libero_full_formal_A_pure_l10

H100; R3/B1; workers {'timan108': 39}; episodes 500; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_full_formal_A_pure_l10.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_full_formal_A_pure_l10.sh
```

## ffk1015_pi05_libero_full_formal_A_pure_spatial

H100; R3/B1; workers {'timan108': 39}; episodes 500; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_full_formal_A_pure_spatial.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_full_formal_A_pure_spatial.sh
```

## ffk1015_pi05_libero_reduced_formal_A_original_off_complete

RTX 4090; R3/B1; workers {'timan107': 64}; episodes 2500; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_reduced_formal_A_original_off_complete.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_reduced_formal_A_original_off_complete.sh
```

## ffk1015_pi05_libero_reduced_formal_A_filtered_levels

RTX 4090; R3/B1; workers {'timan107': 64}; episodes 14000; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_pi05_libero_reduced_formal_A_filtered_levels.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_pi05_libero_reduced_formal_A_filtered_levels.sh
```

## ffk1015_groot_libero_reduced_formal_A_original_off_complete

RTX 4090; R4/B1; workers {'timan107': 64}; episodes 1950; BUILT.

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_groot_libero_reduced_formal_A_original_off_complete.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_groot_libero_reduced_formal_A_original_off_complete.sh
```

# Full-mode second-stage builders

Run each command after that arm's off unit and the same-step pure references are complete. The prepare alias performs this step automatically if needed. The selected cells are fixed at build; an all-qualifying profile becomes a no-op unit.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model dp --mode full --arm original
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model dp --mode full --arm filtered
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model fastwam --mode full --arm original
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model fastwam --mode full --arm filtered
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model cosmos --mode full --arm original
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model cosmos --mode full --arm filtered
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model pi05 --mode full --arm original
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model pi05 --mode full --arm filtered
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model groot --mode full --arm original
```

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model groot --mode full --arm filtered
```
