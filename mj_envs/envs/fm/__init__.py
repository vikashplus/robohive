from gym.envs.registration import register
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Reach to fixed target
register(
    id='FMReachFixed-v0',
    entry_point='mj_envs.envs.fm.base_v0:FMReachEnvFixed',
    max_episode_steps=5000, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/dmanus.mjb',
            'config_path': curr_dir+'/assets/dmanus.config',
            'target_pose': np.array([0, 1, 1, 0, 1, 1, 0, 1, 1]),
        }
)
from mj_envs.envs.arms.franka.reach_v0 import FrankaReachFixed