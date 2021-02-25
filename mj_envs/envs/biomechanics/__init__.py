from gym.envs.registration import register
import os
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))


print("RS:> Registering Biomechanics Envs")
# Finger-tip reaching ==============================
register(id='FingerReachMotorFixed-v0',
            entry_point='mj_envs.envs.biomechanics.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_motorAct_v0.xml',
                'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),}
            }
    )
register(id='FingerReachMotorRandom-v0',
            entry_point='mj_envs.envs.biomechanics.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_motorAct_v0.xml',
                'target_reach_range': {'IFtip': ((0.27, .1, .3), (.1, -.1, .1)),}
            }
    )
register(id='FingerReachMuscleFixed-v0',
            entry_point='mj_envs.envs.biomechanics.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_muscleAct_v0.xml',
                'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),}
            }
    )

register(id='FingerReachMuscleRandom-v0',
            entry_point='mj_envs.envs.biomechanics.reach_v0:ReachEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_muscleAct_v0.xml',
                'target_reach_range': {'IFtip': ((0.27, .1, .3), (.1, -.1, .1)),}
            }
    )


# Finger-Joint posing ==============================
register(id='FingerPoseMotorFixed-v0',
            entry_point='mj_envs.envs.biomechanics.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_motorAct_v0.xml',
                'target_jnt_range': {'IFadb':(0, 0),
                                    'IFmcp':(0, 0),
                                    'IFpip':(.75, .75),
                                    'IFdip':(.75, .75)
                                    },
                'viz_site_targets': ('IFtip',)
            }
    )
register(id='FingerPoseMotorRandom-v0',
            entry_point='mj_envs.envs.biomechanics.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_motorAct_v0.xml',
                'target_jnt_range': {'IFadb':(-.2, .2),
                                    'IFmcp':(-.4, 1),
                                    'IFpip':(.1, 1),
                                    'IFdip':(.1, 1)
                                    },
                'viz_site_targets': ('IFtip',)
            }
    )
register(id='FingerPoseMuscleFixed-v0',
            entry_point='mj_envs.envs.biomechanics.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_muscleAct_v0.xml',
                'target_jnt_range': {'IFadb':(0, 0),
                                    'IFmcp':(0, 0),
                                    'IFpip':(.75, .75),
                                    'IFdip':(.75, .75)
                                    },
                'viz_site_targets': ('IFtip',)
            }
    )
register(id='FingerPoseMuscleRandom-v0',
            entry_point='mj_envs.envs.biomechanics.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/finger/tendon_finger_muscleAct_v0.xml',
                'target_jnt_range': {'IFadb':(-.2, .2),
                                    'IFmcp':(-.4, 1),
                                    'IFpip':(.1, 1),
                                    'IFdip':(.1, 1)
                                    },
                'viz_site_targets': ('IFtip',)
            }
    )

# Hand-Joint posing ==============================
register(id='IFTHPoseMuscleRandom-v0',
            entry_point='mj_envs.envs.biomechanics.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/hand/Index_Thumb_v0.xml',
                'target_jnt_range': {'MCP2_lateral': (-0.349066, 0.349066),
                                    'MCP2_flex': (-0.174533, 1.5708),
                                    'PIP_flex': (-0.0872665, 1.5708),
                                    'DIP_flex': (-0.0872665, 1.5708),
                                    'thumb_abd': (-0.785398, 0.261799),
                                    'thumb_flex': (-0.785398, 1.5708),
                                    'TCP2M_flex': (-0.17, 0.95),
                                    'TCP2M2_flex': (-0.0872665, 1.5708)
                                    },
                'viz_site_targets': ('IFtip','THtip'),
                'normalize_act': False
            }
    )