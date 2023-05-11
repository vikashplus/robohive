# Multi-Task Suite
This suite is designed to study generalization in multi-task settings. RoboHive's multi-task suite builds from `FrankaKitchen` environements originally studied in the [Relay Policy Learning](https://relay-policy-learning.github.io/) project.

# Franka Kitchen
`FrankaKitchen` domain offers a challenging set of manipulation problems in an unstructured environment with many possible tasks to perform. The original set consisted of a franka robot in a kitchen domain. Overtime, Franka Kitchen has became a popular choice of environments for studying multi-task generalization. Its widespread use has led to a few different publically available variations. To help navigate these changes, we name these variations and document its evolution of across various versions below -

## Change log/ History

### FrankaKitchen-v4(RoboHive)
- A part of the RoboHive-v0.5 release. Designed and packaged specifically to study visual generalization.
- Introduces three env variants (fix to control env's default behavior)
    - `Fixed-v4`: State based stationary environments with no randomization
    - `Random-v4`: State based environments with random robot initialization (joint pose + relative position  wrt to kitchen)
    - `Random_v2d-v4`: Visual environment with random robot initialization (joint pose + relative position  wrt to kitchen)
- RoboHive introduces `get_obs()`, `get_prioprio()`, `get_extero()` features. All state envs shoud used `obs`. All visual envs should use `proprio` and `extero` data.
- Init robot state recovered from([kitchen_demos_multitask-v0](https://github.com/google-research/relay-policy-learning/blob/master/kitchen_demos_multitask.zip))
- Four explicit cameras introduced to aid multi-view generalization studies -- `top`, `left`, `right`, `wrist`
- Grouping environments into sets to help with results
    - `FK1_FIXED_20` = `FK1_FIXED_5A`+`FK1_FIXED_5B`+`FK1_FIXED_5C`+`FK1_FIXED_5D`
    - `FK1_RANDOM_20` = `FK1_RANDOM_5A`+`FK1_RANDOM_5B`+`FK1_RANDOM_5C`+`FK1_RANDOM_5D`
    - `FK1_RANDOM_V2D_20` = `FK1_RANDOM_V2D_5A`+`FK1_RANDOM_V2D_5B`+`FK1_RANDOM_V2D_5C`+`FK1_RANDOM_V2D_5D`


### FrankaKitchen-v3(RoboHive+R3M) &rarr; _Depricated_
- A part of [RoboHive-V0.0.5](https://github.com/vikashplus/robohive/releases/tag/v0.0.5) pre-release that was used for the [R3M project](https://sites.google.com/view/robot-r3m/) to study visual generalization in FrankaKitchen domain.
- To study visual generalization creates and introduces a new dataset [R3M(bc_data_clean)](https://github.com/facebookresearch/r3m/tree/eval/evaluation#downloading-demonstration-data). This dataset is generated using successful pretrained state based single task policies, and is of fairly high quality.
- R3M uses [v3(R3M)](https://github.com/vikashplus/robohive/blob/5a8cb3944824abe155efe9bcaf110c46c19c5564/robohive/envs/relay_kitchen/__init__.py#L145-L149) envs
    - [issue] as its not easy (yet) to programatically pick env entrypoints, R3M [overrides](https://github.com/vikashplus/robohive/commit/5a8cb3944824abe155efe9bcaf110c46c19c5564) the default behavior of the environments from a fixed kitchen to one where the kitchen randomly moves around, making the tasks visually more challenging and ready for visual generalization studies.
    - [issue] The default behavior of the env is changed, but the envs are still named as `-v3`
    - [issue] R3M uses 3 camera views. Two of which are names cameras. The third however is the `default` camera of the environment. Default cameras behavior changes acorss commits making it hard to reproduce results.
- Env_details
    - Observations:

### FrankaKitchen-v3(RoboHive) &rarr; _Depricated_
- A part of [RoboHive-V0.1](https://github.com/vikashplus/robohive/releases/tag/v0.1) pre-release
- Builds off directly from the original _FrankaKitchen (Realy Policy Learning)_ release. Creates a few internal versions
    - [v1](https://github.com/vikashplus/robohive/blob/5a8cb3944824abe155efe9bcaf110c46c19c5564/robohive/envs/relay_kitchen/__init__.py#L17-L22)- for porting envs to RoboHive ecosystem.
        - [Issue] called v0 in codebase
    - [v2](https://github.com/vikashplus/robohive/blob/5a8cb3944824abe155efe9bcaf110c46c19c5564/robohive/envs/relay_kitchen/__init__.py#L24-L142)- for making them ready for multi-task studies.
        - [Issue] uses observation that are task specific, therefore not ideal for multi-task generalizaiton
        - [Issue] introduces a few different entry points to control different kitchen and arms behaviors at reset - KitchenFrankaFixed, KitchenFrankaDemo, KitchenFrankaRandom, KitchenFrankaRandomDesk. Picking entry point requires manaul edits
        - [Issue] All entrypoints are called with the same env name making it hard to distinguish different env variants
    - [v3](https://github.com/vikashplus/robohive/blob/5a8cb3944824abe155efe9bcaf110c46c19c5564/robohive/envs/relay_kitchen/__init__.py#L145-L149)- makes it ready for multi-task settings by standardising observations.
        - [Issue] Introduces a few different entry points to control different kitchen and arms behaviors at reset - KitchenFrankaFixed, KitchenFrankaDemo, KitchenFrankaRandom, KitchenFrankaRandomDesk. Picking entry point requires manaul edits
        - [Issue] All entrypoints are called with the same env name making it hard to distinguish different env variants

### FrankaKitchen (D4RL)
- A close adaptation of the original Adept Models+Envs from _FrankaKitchen (Realy Policy Learning)_ that was packaged in [D4RL](https://sites.google.com/view/d4rl/home) project to study offline RL.
- A small dataset was curated from the original ([kitchen_demos_multitask-v0](https://github.com/google-research/relay-policy-learning/blob/master/kitchen_demos_multitask.zip)) dataset to study offline reinforcement learning. This subset was made available as a part fo the [D4RL release](http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/)
    - [issue] Picks a subset of demonstrations. Ignores the richness presented by the [kitchen_demos_multitask-v0](https://github.com/google-research/relay-policy-learning/blob/master/kitchen_demos_multitask.zip) dataset.
- Env details: Adds a [wrapper](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/kitchen/kitchen_envs.py) around the original base class to support D4RL features
    - [issues] Picks a small env subset. Ignores the richness that FrankaKitchen domain presented for multi-task studies

### FrankaKitchen (Realy Policy Learning) &rarr; _Depricated_
- Original simulations and envs [Adept Models+Envs](https://github.com/google-research/relay-policy-learning) developed by Vikash Kumar and Michael Wu.
- [Relay Policy Learning](https://relay-policy-learning.github.io/) project studied these environments. The version studied for this project was open sourced as [kitchen_relax](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/__init__.py#L21) environment.
- Human dataset ([kitchen_demos_multitask-v0](https://github.com/google-research/relay-policy-learning/blob/master/kitchen_demos_multitask.zip)) collected for [Relay Policy Learning](https://relay-policy-learning.github.io/) project was also released as part of the project
- Env details ([base class](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/kitchen_multitask_v0.py#L130)): Observations: [robot_qp, obj_q, goal](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/kitchen_multitask_v0.py#L130), Init_qpos: Picked from human demos for the task [microwave kettle slide hinge](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/kitchen_multitask_v0.py#L62)