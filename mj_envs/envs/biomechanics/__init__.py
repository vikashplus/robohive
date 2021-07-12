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
                'model_path': curr_dir+'/../../sims/neuromuscular_sim/hand/Index_Thumb_v0.xml',
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
                'normalize_act': False,
                'reset_type': 'none',           # none, init, random
                'target_type': 'generate',      # switch / generate
            }
    )

register(id='HandPoseAMuscleFixed-v0',
            entry_point='mj_envs.envs.biomechanics.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/assets/hand/2nd_hand_pose.xml',
                'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
                'target_jnt_value': np.array([0, 0, 0, -0.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]),
                'normalize_act': False,
                'reset_type': "random",        # none, init, random
                'target_type': 'fixed',      # switch / generate/ fixed
            }
    )

# Hand-Joint key turn ==============================
register(id='IFTHKeyTurnFixed-v0',
            entry_point='mj_envs.envs.biomechanics.key_turn_v0:KeyTurnFixedEnvV0',
            max_episode_steps=200,
            kwargs={
                'model_path': curr_dir+'/assets/hand/Index_Thumb_keyturn_v0.xml',
                'normalize_act': False
            }
    )

register(id='IFTHKeyTurnRandom-v0',
            entry_point='mj_envs.envs.biomechanics.key_turn_v0:KeyTurnRandomEnvV0',
            max_episode_steps=200,
            kwargs={
                'model_path': curr_dir+'/assets/hand/Index_Thumb_keyturn_v0.xml',
                'normalize_act': False
            }
    )


# Hold objects ==============================
register(id='HandObjHoldFixed-v0',
            entry_point='mj_envs.envs.biomechanics.obj_hold_v0:ObjHoldFixedEnvV0',
            max_episode_steps=75,
            kwargs={
                'model_path': curr_dir+'/assets/hand/2nd_hand_hold.xml',
                'normalize_act': False
            }
    )

register(id='HandObjHoldRandom-v0',
            entry_point='mj_envs.envs.biomechanics.obj_hold_v0:ObjHoldRandomEnvV0',
            max_episode_steps=75,
            kwargs={
                'model_path': curr_dir+'/assets/hand/2nd_hand_hold.xml',
                'normalize_act': False
            }
    )


# Pen twirl ==============================
register(id='HandPenTwirlFixed-v0',
            entry_point='mj_envs.envs.biomechanics.pen_v0:PenTwirlFixedEnvV0',
            max_episode_steps=50,
            kwargs={
                'frame_skip': 5,
                'model_path': curr_dir+'/assets/hand/2nd_hand_pen.xml',
                'normalize_act': True
            }
    )

register(id='HandPenTwirlRandom-v0',
            entry_point='mj_envs.envs.biomechanics.pen_v0:PenTwirlRandomEnvV0',
            max_episode_steps=50,
            kwargs={
                'frame_skip': 5,
                'model_path': curr_dir+'/assets/hand/2nd_hand_pen.xml',
                'normalize_act': True
            }
    )

# Baoding ==============================
register(id='BaodingFixed-v1',
            entry_point='mj_envs.envs.biomechanics.baoding_v1:BaodingFixedEnvV1',
            max_episode_steps=200,
            kwargs={
                'model_path': curr_dir+'/assets/hand/2nd_hand_baoding.xml',
                'normalize_act': True,
                'reward_option':0
            }
    )
register(id='BaodingFixed4th-v1',
            entry_point='mj_envs.envs.biomechanics.baoding_v1:BaodingFixedEnvV1',
            max_episode_steps=200,
            kwargs={
                'model_path': curr_dir+'/assets/hand/2nd_hand_baoding.xml',
                'normalize_act': True,
                'reward_option':1
            }
    )
register(id='BaodingFixed8th-v1',
            entry_point='mj_envs.envs.biomechanics.baoding_v1:BaodingFixedEnvV1',
            max_episode_steps=200,
            kwargs={
                'model_path': curr_dir+'/assets/hand/2nd_hand_baoding.xml',
                'normalize_act': True,
                'reward_option':2
            }
    )