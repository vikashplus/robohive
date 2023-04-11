""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gym.envs.registration import register
from robohive.envs.env_variants import register_env_variant
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Arms Envs")

# FRANKA REACH =======================================================================
from robohive.envs.arms.reach_base_v0 import ReachBaseV0

# Reach to fixed target
register(
    id='FrankaReachFixed-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_reach_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_reach_v0.config',
        'robot_site_name': "end_effector",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.2, 0.3, 1.2], 'low':[0.2, 0.3, 1.2]}
    }
)

# Reach to random target
register(
    id='FrankaReachRandom-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_reach_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_reach_v0.config',
        'robot_site_name': "end_effector",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.3, .5, 1.2], 'low':[-.3, .1, .8]}
    }
)

# Reach to random target using visual inputs
def register_visual_envs(encoder_type):
    register_env_variant(
        env_id='FrankaReachRandom-v0',
        variant_id='FrankaReachRandom_v{}-v0'.format(encoder_type),
        variants={'proprio_keys':
                    ['qp', 'qv'],
                'visual_keys':[
                    "rgb:left_cam:224x224:{}".format(encoder_type),
                    "rgb:right_cam:224x224:{}".format(encoder_type),
                    "rgb:top_cam:224x224:{}".format(encoder_type)]
        },
        silent=True
    )
for enc in ["r3m18", "r3m34", "r3m50", "rrl18", "rrl34", "rrl50", "2d"]:
    register_visual_envs(enc)



# FRANKA PUSH =======================================================================
from robohive.envs.arms.push_base_v0 import PushBaseV0

# Push object to target
register(
    id='FrankaPushFixed-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_ycb_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_ycb_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.5, 0.4, 0.78], 'low':[0.5, 0.4, 0.78]}
    }
)

# Push object to target
register(
    id='FrankaPushRandom-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_ycb_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_ycb_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.5, 0.4, 0.78], 'low':[0.4, -0.4, 0.78]}
    }
)

register(
    # Init position: [0.51,0.23,0.98]
    # Init euler: [-np.pi/4+0.3, np.pi+0.3, -3*np.pi/4,]
    id='FrankaBinPush-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.52, 0.126, 0.955], 'low':[0.48, 0.125, 0.945]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.5, -0.22, 1.085], 'low':[0.5, -0.22, 1.085]},
        'pos_limits': {'eef_low': [0.315, -0.3, 0.89, -0.485, 3.14, -2.36, 0.4],
                       'eef_high': [0.695, 0.275, 1.175, -0.485, 3.14, -2.36, 0.4]
                       },
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.534, 0.401, 0.0, -1.971, -0.457, 0.490, 1.617, 0.4, 0.4],
    }
)

register(
    id='FrankaBinPushTeleop-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_push_teleop_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.52, 0.126, 0.955], 'low':[0.48, 0.125, 0.945]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.22, 0.5, 1.02], 'low':[0.22, 0.5, 1.02]},
        'pos_limits': {'eef_low': [0.2, 0.05, 0.9, -0.485, 3.14, -2.36, 0.3507],
                       'eef_high': [0.7, 0.6, 1.1, -0.485, 3.14, -2.36, 0.3507]
                       },                      
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.767,0.492,-0.064,-1.965,-0.358,0.535,1.713,0.349,0.349],
    }
)

register(
    id='FrankaBinPushReal-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.52, 0.126, 0.955], 'low':[0.48, 0.125, 0.945]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.22, 0.5, 1.02], 'low':[0.22, 0.5, 1.02]},
        'pos_limits': {'eef_low': [0.2, 0.05, 0.9, -0.485, 3.14, -2.36, 0.3507],
                       'eef_high': [0.7, 0.6, 1.1, -0.485, 3.14, -2.36, 0.3507]
                       },                      
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.767,0.492,-0.064,-1.965,-0.358,0.535,1.713,0.349,0.349],
    }
)

register(
    # Init position: [0.36, 0.0,  1.34]
    # Init euler: [3.14, 0.5, -0.8,]
    id='FrankaHangPush-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_hang_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_hang_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.475, 0.025, 1.31], 'low':[0.425, -0.025, 1.3]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.621, 0.0, 1.333], 'low':[0.621, 0.0, 1.333]},
        'pos_limits': {'eef_low': [0.3, -0.1, 1.25, 3.14, 0.5, -0.8, 0.2],
                       'eef_high': [0.8, 0.1, 1.5, 3.14, 0.5, -0.8, 0.2]
                       },
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.0227, -0.9291, -0.014, -2.5568, -0.029, 0.5052, -0.783, 0.2, 0.2],
    }
)

register(
    # Init position: [0.36, 0.0,  1.34]
    # Init euler: [3.14, 0.5, -0.8,]
    id='FrankaPlanarPush-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_planar_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_planar_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.52, 0.31, 0.855], 'low':[0.44, 0.25, 0.845]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.48, -0.1, 0.845], 'low':[0.48, -0.1, 0.845]},
        'pos_limits': {'eef_low': [0.3, -0.4, 0.865, 3.14, 0.0, 0.28, 0.2],
                       'eef_high': [0.8, 0.4, 0.965, 3.14, 0.0, 1.28, 0.2]
                       },
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.665, 0.567,  0.0, -1.999, 0.0,  0.9172, 1.4541, 0.2, 0.2],
    }
)

register(
    # Init position: [0.36, 0.0,  1.34]
    # Init euler: [3.14, 0.5, -0.8,]
    id='FrankaPlanarPushTeleop-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_planar_push_teleop_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_planar_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.52, -0.31, 0.855], 'low':[0.44, -0.25, 0.845]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.48, -0.1, 0.845], 'low':[0.48, -0.1, 0.845]},
        'pos_limits': {'eef_low': [0.2, -0.2, 0.9, 3.14, 0.0, 0.28, 0.0],
                       'eef_high': [0.7, 0.5, 0.95, 3.14, 0.0, 1.28, 0.0]
                       },
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.467,0.309,0.088,-2.344,-0.145,1.054,1.4,0.0,0.0],
    }
)

register(
    # Init position: [0.36, 0.0,  1.34]
    # Init euler: [3.14, 0.5, -0.8,]
    id='FrankaPlanarPushReal-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_planar_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_planar_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'obj_pos_limits': {'high':[0.52, 0.31, 0.855], 'low':[0.44, 0.25, 0.845]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.48, -0.1, 0.845], 'low':[0.48, -0.1, 0.845]},
        'pos_limits': {'eef_low': [0.2, -0.2, 0.9, 3.14, 0.0, 0.28, 0.0],
                       'eef_high': [0.7, 0.5, 0.95, 3.14, 0.0, 1.28, 0.0]
                       },
        'vel_limits': {'eef':[0.15, 0.15, 0.15],
                        'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 1.0]
                        },
        'init_qpos': [0.467,0.309,0.088,-2.344,-0.145,1.054,1.4,0.0,0.0],
    }
)

# Reach to random target using visual inputs
def register_push_visual_envs(env_name, encoder_type, cams, real=False, real_cams=None):
    proprio_keys = ['qp', 'qv', 'grasp_pos', 'grasp_rot']
    visual_keys = []
    assert((not real and real_cams is None) or (real and real_cams is not None))
    if not real:
        for cam in cams:
            visual_keys.append('rgb:'+cam+':224x224:{}'.format(encoder_type))
            visual_keys.append('d:'+cam+':224x224:{}'.format(encoder_type))
    else:
        for cam in cams:
            visual_keys.append('rgb:'+cam+':240x424:{}'.format(encoder_type))
            visual_keys.append('d:'+cam+':240x424:{}'.format(encoder_type))      
        for cam in real_cams:
            visual_keys.append('rgb:'+cam+':240x424:{}'.format(encoder_type))
            visual_keys.append('d:'+cam+':240x424:{}'.format(encoder_type))              
       
    register_env_variant(
        env_id='{}-v0'.format(env_name),
        variant_id='{}_v{}-v0'.format(env_name, encoder_type),
        variants={'proprio_keys': proprio_keys,
                  'visual_keys': visual_keys
        },
        silent=True
    )

push_cams =  ['top_cam', 'Franka_wrist_cam']
real_cams = ['left_cam', 'right_cam']
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_push_visual_envs('FrankaBinPush', enc, cams=push_cams)
    register_push_visual_envs('FrankaBinPushTeleop', enc, cams=push_cams, real_cams=real_cams, real=True)
    register_push_visual_envs('FrankaBinPushReal', enc, cams=push_cams, real_cams=real_cams, real=True)

for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_push_visual_envs('FrankaHangPush', enc, cams=push_cams)

planar_push_cams = ['right_cam', 'Franka_wrist_cam']
planar_push_real_cams = ['top_cam', 'left_cam']
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_push_visual_envs('FrankaPlanarPush', enc, cams=planar_push_cams)
    register_push_visual_envs('FrankaPlanarPushTeleop', enc, cams=planar_push_cams, real_cams=planar_push_real_cams, real=True)
    register_push_visual_envs('FrankaPlanarPushReal', enc, cams=planar_push_cams, real_cams=planar_push_real_cams, real=True)

# FRANKA PICK =======================================================================
register(
    id='FrankaPickPlaceFixed-v0',
    entry_point='robohive.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_busbin_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'target_xyz_range': {'high':[-.235, 0.5, 0.85], 'low':[-.235, 0.5, 0.85]},
    }
)
register(
    id='FrankaPickPlaceRandom-v0',
    entry_point='robohive.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_busbin_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[-.135, 0.6, 0.85], 'low':[-.335, 0.4, 0.85]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]}
    }
)

register(
    id='FrankaBinPick-v0',
    entry_point='robohive.envs.arms.bin_pick_v0:BinPickV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_pick_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_pick_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.5, 0.0, 1.1], 'low':[0.5, 0.0, 1.1]}
    }

)

register(
    id='FrankaBinPickReal-v0',
    entry_point='robohive.envs.arms.bin_pick_v0:BinPickV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_pick_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_pick_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.5, 0.0, 1.1], 'low':[0.5, 0.0, 1.1]},
        'init_qpos': [0., 0., 0., -1.7359, 0., 0.2277,  -1.57, 0.0]
    }
)

register(
    id='FrankaBinPickRealRP03-v0',
    entry_point='robohive.envs.arms.bin_pick_v0:BinPickV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_pick_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_pick_rp03_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.5, 0.0, 1.1], 'low':[0.5, 0.0, 1.1]},
        'init_qpos': [0., 0., 0., -1.7359, 0., 0.2277,  -1.57, 0.0]
    }
)
# Reach to random target using visual inputs
def register_bin_pick_visual_envs(env_name, encoder_type, real=False):
    proprio_keys = ['qp', 'qv', 'grasp_pos', 'grasp_rot']

    if not real:
        visual_keys = ["rgb:left_cam:224x224:{}".format(encoder_type),
                       "d:left_cam:224x224:{}".format(encoder_type),
                       "rgb:right_cam:224x224:{}".format(encoder_type),
                       "d:right_cam:224x224:{}".format(encoder_type)]
    else:
        visual_keys = ["rgb:left_cam:240x424:{}".format(encoder_type),
                       "d:left_cam:240x424:{}".format(encoder_type),
                       "rgb:right_cam:240x424:{}".format(encoder_type),
                       "d:right_cam:240x424:{}".format(encoder_type),
                       "rgb:top_cam:240x424:{}".format(encoder_type),
                       "d:top_cam:240x424:{}".format(encoder_type),
                       "rgb:Franka_wrist_cam:240x424:{}".format(encoder_type),
                       "d:Franka_wrist_cam:240x424:{}".format(encoder_type)]

    register_env_variant(
        env_id='{}-v0'.format(env_name),
        variant_id='{}_v{}-v0'.format(env_name, encoder_type),
        variants={'proprio_keys': proprio_keys,
                  'visual_keys':visual_keys
        },
        silent=True
    )
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_bin_pick_visual_envs('FrankaBinPick', enc)
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_bin_pick_visual_envs('FrankaBinPickReal', enc, real=True)    
    register_bin_pick_visual_envs('FrankaBinPickRealRP03', enc, real=True)    

# REORIENT ====================================================================

register(
    id='FrankaBinReorient-v0',
    entry_point='robohive.envs.arms.bin_reorient_v0:BinReorientV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_dmanus_reorient_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_dmanus_reorient_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.5, 0.0, 1.1], 'low':[0.5, 0.0, 1.1]}
    }

)

# FETCH =======================================================================
from robohive.envs.arms.reach_base_v0 import ReachBaseV0

# Reach to fixed target
register(
    id='FetchReachFixed-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/fetch/assets/fetch_reach_v0.xml',
        'config_path': curr_dir+'/fetch/assets/fetch_reach_v0.config',
        'robot_site_name': "grip",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.2, 0.3, 1.2], 'low':[0.2, 0.3, 1.2]}
    }
)

# Reach to random target
register(
    id='FetchReachRandom-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/fetch/assets/fetch_reach_v0.xml',
        'config_path': curr_dir+'/fetch/assets/fetch_reach_v0.config',
        'robot_site_name': "grip",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.3, .5, 1.2], 'low':[-.3, .1, .8]}
    }
)
