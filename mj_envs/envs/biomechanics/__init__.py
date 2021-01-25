from gym.envs.registration import register
import os
import numpy as np


# Finger-tip reach ==============================
register(id='FingerReachMotorFixed-v0',
            entry_point='mj_envs.envs.biomechanics.finger.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'motor',
                'reach_target_xyz_range': np.array(([0.2, 0.05, 0.20], [0.2, 0.05, 0.20]))
            }
    )
register(id='FingerReachMotorRandom-v0',
            entry_point='mj_envs.envs.biomechanics.finger.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'motor',
                'reach_target_xyz_range': np.array(([0.27, .1, .3], [.1, -.1, .1]))
            }
    )
register(id='FingerReachMuscleFixed-v0',
            entry_point='mj_envs.envs.biomechanics.finger.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'muscle',
                'reach_target_xyz_range': np.array(([0.2, 0.05, 0.20], [0.2, 0.05, 0.20]))
            }
    )
register(id='FingerReachMuscleRandom-v0',
            entry_point='mj_envs.envs.biomechanics.finger.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'muscle',
                'reach_target_xyz_range': np.array(([0.27, .1, .3], [.1, -.1, .1]))
            }
    )

# Joint pose reach ==============================
register(id='FingerPoseMotorFixed-v0',
            entry_point='mj_envs.envs.biomechanics.finger.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'motor',
                'pose_target_xyz_range': np.array(([0, 0, .75, .75], [0, 0, .75, .75]))
            }
    )
register(id='FingerPoseMotorRandom-v0',
            entry_point='mj_envs.envs.biomechanics.finger.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'motor',
                'pose_target_jnt_range': np.array(([-.4, -.2, .1, .1], [1, .2, 1, 1]))
            }
    )
register(id='FingerPoseMuscleFixed-v0',
            entry_point='mj_envs.envs.biomechanics.finger.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'muscle',
                'pose_target_xyz_range': np.array(([0, 0, .75, .75], [0, 0, .75, .75]))
            }
    )
register(id='FingerPoseMuscleRandom-v0',
            entry_point='mj_envs.envs.biomechanics.finger.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'act_mode': 'muscle',
                'pose_target_jnt_range': np.array(([-.4, -.2, .1, .1], [1, .2, 1, 1]))
            }
    )