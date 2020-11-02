from gym.envs.registration import register
# from mjrl.envs.mujoco_env import MujocoEnv

# Kithen
register(
    id='kitchen-v0',
    entry_point='mj_envs.envs.relay_kitchen:KitchenTasksV0',
    max_episode_steps=280,
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v1 import KitchenTasksV0