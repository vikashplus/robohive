# Replay Human Demos on Adept_envs vs. RoboHive envs

## Issue
After parsing the original Relay human play data via the official instructions provided [here](https://github.com/google-research/relay-policy-learning) and obtaining a series of pickle files, replaying this pickle file on the original `kitchen_relax-v1` (from adept_envs) gives different behaviors than replaying it on `FK1_RelaxFixed-v4` (from robohive). 

Visually, replaying the pickle file on `kitchen_relax-v1` gives visually perfect demonstrations, while replaying on `FK1_RelaxFixed-v4` gives unsuccessful and aggressive behavior.

The qpos after calling `step` for even once is different across the two environments.

## Steps to recreate
1. (*no need to perform*) Parse the original Relay human play data from  `.mjl` files to `.pkl` files. Here I attached one such `.pkl` file which can be directly used for the following steps.
2. In your relay repo, change `self.robot_noise_ratio` from 0.1 to 0 on [this line](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/kitchen_multitask_v0.py#L43) to make sure the correct qpos when replaying on `kitchen_relax-v1` can be exactly recreated across runs. 
3. Checkout `data-gen` branch (or just download the [debug]() folder and put it in your local robohive repo. Everything needed should be in this folder).
4. In side the `./debug` folder, run 
    ```
    sim_backend=MUJOCO python relay_debug.py -e <env> -d kitchen_relax-v10_path.pkl -sp True
    ```
    for replaying a human trajectory on `<env>`. Should run this for both `kitchen_relax-v1` and `FK1_RelaxFixed-v4` and compare. 

    *Note: the function is implemented based on the logic in the parsing script in [here](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/utils/parse_demos.py#L84) and [here](https://github.com/vikashplus/mjrl_dev/blob/redesign/mjrl_dev/datasets/franka_kitchen/utils/parse_demos.py#L86).*
5. The script prints qpos before and after the environment calls the `step` function. You can see that the qpos before was the same in both cases, but diverged after calling `step`. You can also comment out the pdb I added in this script to render the full trajectory.




## Potential Cause?
There might be some mismatch in the simulated robot's `step` function of both envs? 
- [robohive's step function](https://github.com/vikashplus/robohive/blob/main/robohive/robot/robot.py#L659)
- [relay's step function](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/robot/franka_robot.py#L178)