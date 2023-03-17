# Robohive Hands on totorial (v0.2)

## Installations
Follow instructions [here](https://github.com/vikashplus/mj_envs/tree/v0.2dev)

## Examine envs

- Visualize
```
python mj_envs/utils/examine_env.py -e FrankaReachRandom-v0
```
- Visualize with policy

```
python utils/examine_env.py -e kitchen_sdoor_open-v3 -p  /Users/vikashplus/Projects/mj_envs/kitchen/outputs_kitchenJ5d_3.9/kitchen_sdoor_open-v3/2022-03-09_11-45-28/19_env=kitchen_sdoor_open-v3,seed=2/iterations/best_policy.pickle
```
- Render offscreen
```
python utils/examine_env.py -e FrankaReachRandom-v0 -r offscreen -n 1
```

- Render paths
```
python utils/examine_env.py -e FrankaReachRandom-v0 -r offscreen -n 1 --plot_paths True
```

- Inspect paths (add a breakpoint [here](https://github.com/vikashplus/mj_envs/blob/v0.2dev/mj_envs/utils/examine_env.py#L86) to inspect)

## Pre-Releases
Releases: https://github.com/vikashplus/mj_envs/tags

Saved Baselines: https://github.com/vikashplus/mj_envs/releases/tag/v0.1

## Repo structure
- envs
- sims
- robot
- utils

## Create Envs
- env_base
- Obs and rewards definitions

## Create multi-task envs
- subtasks [examples](https://github.com/vikashplus/mj_envs/tree/v0.2dev/mj_envs/envs/multi_task)


