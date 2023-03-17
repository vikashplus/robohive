from gym.envs.registration import register
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Reach to fixed target
# register(
#     id='DManusReachFixed-v0',
#     entry_point='robohive.envs.fm.franka_ee_pose_v0:FMReachEnvFixed',
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
    entry_point='robohive.envs.fm.franka_ee_pose_v0:FrankaEEPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus.xml',
            'config_path': curr_dir+'/assets/franka_dmanus.config',
            'target_pose': np.array([0, 0, 0, -1.57, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]),
        }
)

# Pose to random target
register(
    id='rpFrankaDmanusPoseRandom-v0',
    entry_point='robohive.envs.fm.franka_ee_pose_v0:FrankaEEPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus.xml',
            'config_path': curr_dir+'/assets/franka_dmanus.config',
            'target_pose': 'random'
        }
)


# Pose to fixed target
register(
    id='rpFrankaRobotiqPoseFixed-v0',
    entry_point='robohive.envs.fm.franka_ee_pose_v0:FrankaRobotiqPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_robotiq.xml',
            'config_path': curr_dir+'/assets/franka_robotiq.config',
            'target_pose': np.array([0, 0, 0, -1.57, 0, 0, 0, 0]),
        }
)

# Pose to fixed target
register(
    id='rpFrankaRobotiqPoseRandom-v0',
    entry_point='robohive.envs.fm.franka_ee_pose_v0:FrankaRobotiqPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_robotiq.xml',
            'config_path': curr_dir+'/assets/franka_robotiq.config',
            'target_pose': 'random',
        }
)

# Pose to fixed target
encoder_type = "2d"
# img_res="480x640"
img_res="240x424"
register(
    id='rpFrankaRobotiqData-v0',
    entry_point='robohive.envs.fm.franka_robotiq_data_v0:FrankaRobotiqData',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': '/assets/franka_robotiq.xml',
        'config_path': curr_dir+'/assets/franka_robotiq.config',
        'nq_arm':7,
        'nq_ee':1,
        'name_ee':'end_effector',
        'visual_keys':[
            # customize the visual keys
            "rgb:left_cam:{}:{}".format(img_res, encoder_type),
            "rgb:right_cam:{}:{}".format(img_res, encoder_type),
            "rgb:top_cam:{}:{}".format(img_res, encoder_type),
            "rgb:Franka_wrist_cam:{}:{}".format(img_res, encoder_type),
            "d:left_cam:{}:{}".format(img_res, encoder_type),
            "d:right_cam:{}:{}".format(img_res, encoder_type),
            "d:top_cam:{}:{}".format(img_res, encoder_type),
            "d:Franka_wrist_cam:{}:{}".format(img_res, encoder_type),
            ]
        }
)