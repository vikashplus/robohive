from gym.envs.registration import register
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Reach to fixed target
# register(
#     id='DManusReachFixed-v0',
#     entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FMReachEnvFixed',
#     max_episode_steps=50, #50steps*40Skip*2ms = 4s
#     kwargs={
#             'model_path': '/assets/dmanus.xml',
#             'config_path': curr_dir+'/assets/dmanus.config',
#             'target_pose': np.array([0, 1, 1, 0, 1, 1, 0, 1, 1]),
#         }
# )

# Pose to fixed target
register(
    id='rpFrankaDmanusPoseFixed-v0',
    entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FrankaDmanusPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus.xml',
            'config_path': curr_dir+'/assets/franka_dmanus.config',
            'target_pose': np.array([0, 0, 0, -1.57, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
        }
)

# Pose to random target
register(
    id='rpFrankaDmanusPoseRandom-v0',
    entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FrankaDmanusPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus.xml',
            'config_path': curr_dir+'/assets/franka_dmanus.config',
            'target_pose': 'random'
        }
)
