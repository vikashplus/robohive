from gym.envs.registration import register
import os
import numpy as np


# Reach to fixed target
register(id='BallReachFixed-v0',
            entry_point='mj_envs.envs.touch.ballonplate.reach_v0:BallOnPlateReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'target_xy_range': np.array(([0.0, 0.0], [0.0, 0.0])),
                'ball_xy_range': np.array(([-.05, -.05], [.05, .05]))
            }
    )

# Reach to random target
register(id='BallReachRandom-v0',
            entry_point='mj_envs.envs.touch.ballonplate.reach_v0:BallOnPlateReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'target_xy_range': np.array(([-.05, -.05], [.05, .05])),
                'ball_xy_range': np.array(([-.05, -.05], [.05, .05]))
            }
    )

# Reach to random target using touch inputs
register(id='BallReachRandom_touch10-v0',
            entry_point='mj_envs.envs.touch.ballonplate.reach_v0:BallOnPlateReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'target_xy_range': np.array(([-.05, -.05], [.05, .05])),
                'ball_xy_range': np.array(([-.05, -.05], [.05, .05])),
                'obs_keys': ['plate', 'dplate', 'touch', 'target_err'],
                'tac_n': 10,
                'tac_r': 0.008
            }
    )

# Reach to random target using touch inputs
register(id='BallReachRandom_touch20-v0',
            entry_point='mj_envs.envs.touch.ballonplate.reach_v0:BallOnPlateReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'target_xy_range': np.array(([-.05, -.05], [.05, .05])),
                'ball_xy_range': np.array(([-.05, -.05], [.05, .05])),
                'obs_keys': ['plate', 'dplate', 'touch', 'target_err'],
                'tac_n': 20,
                'tac_r': 0.004
            }
    )