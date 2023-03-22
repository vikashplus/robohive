""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gym.envs.registration import register
from mj_envs.envs.env_variants import register_env_variant
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RS:> Registering Arms Envs")

# FRANKA REACH =======================================================================
from mj_envs.envs.arms.reach_base_v0 import ReachBaseV0

# Reach to fixed target
register(
    id='FrankaReachFixed-v0',
    entry_point='mj_envs.envs.arms.reach_base_v0:ReachBaseV0',
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
    entry_point='mj_envs.envs.arms.reach_base_v0:ReachBaseV0',
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
def register_reacher_visual_envs(encoder_type):
    register_env_variant(
        env_id='FrankaReachRandom-v0',
        variant_id='FrankaReachRandom_v{}-v0'.format(encoder_type),
        variants={'obs_keys':
                    ['qp', 'qv',
                    "rgb:left_cam:224x224:{}".format(encoder_type),
                    "rgb:right_cam:224x224:{}".format(encoder_type),
                    "rgb:top_cam:224x224:{}".format(encoder_type)]
        },
        silent=True
    )
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_reacher_visual_envs(enc)



# FRANKA PUSH =======================================================================
from mj_envs.envs.arms.push_base_v0 import PushBaseV0

# Push object to target
register(
    id='FrankaPushFixed-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_ycb_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_ycb_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[-.4, 0.5, 0.78], 'low':[-.4, 0.5, 0.78]}
    }
)

# Push object to target
register(
    id='FrankaPushRandom-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_ycb_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_ycb_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.4, 0.5, 0.78], 'low':[-.4, .4, 0.78]}
    }
)

register(
    # Init position: [-0.19,0.48,0.88]
    # Init euler: [3.14, 0.5, 0.0]
    id='FrankaBinPush-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'object_init_perturb': {'high':[0.0, 0.02, 0.01], 'low':[-0.01, -0.02, 0.0]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.22, 0.5, 1.02], 'low':[0.22, 0.5, 1.02]},
        'init_qpos': [0.4653, 0.5063, 0.0228, -2.1195, -0.6052, 0.7064, 2.5362, 0.025, 0.025],
        'pos_limit_low':[-0.275, 0.315, 0.825, 3.14, 0.5, 0.0, 0.025],
        'pos_limit_high':[0.22, 0.695, 1.06, 3.14, 0.5, 0.0, 0.025],
        'success_mask': {'left':0, 'right': 224, 'top':0, 'bottom':224, 'thresh':0.2}
    }
)

register(
    # Init position: [-0.19,0.48,0.88]
    # Init euler: [3.14, 0.5, 0.0]
    id='FrankaBinPushReal-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_bin_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_bin_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'object_init_perturb': {'high':[0.0, 0.02, 0.01], 'low':[-0.01, -0.02, 0.0]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.22, 0.5, 1.02], 'low':[0.22, 0.5, 1.02]},
        'init_qpos': [0.833,0.511,-0.131,-1.891,-0.318,0.601,1.75, 0.0168, 0.0168],
        'pos_limit_low':[-0.6,0.2,0.9, 3.14, 0.5, -0.66, 0.0168],
        'pos_limit_high':[-0.05,0.7,1.1, 3.14, 0.5, -0.66, 0.0168],
        'success_mask': {'left':105, 'right': 122, 'top':120, 'bottom':138, 'thresh':0.5}
    }
)

register(
    # Init position: [0.0142 0.3584 1.2545]
    # Init euler: [0.35,3.14,1.57]
    id='FrankaHangPush-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_hang_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_hang_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'object_init_perturb': {'high':[0.01, 0.01, 0.01], 'low':[-0.01, -0.01, 0.0]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.0, 0.621, 1.333], 'low':[0.0, 0.621, 1.333]},
        'init_qpos': [0.0366, -0.9403, -0.0278, -2.7104, -0.0422,  0.5814,  0.0284, 0.025, 0.025],
        'pos_limit_low': [-0.1, 0.3, 1.25, 3.14,0.35,-1.57, 0.025],
        'pos_limit_high': [0.1, 0.8, 1.5, 3.14,0.35,-1.57, 0.025],
        'success_mask': {'left':100, 'right': 156, 'top':100, 'bottom':156}
    }
)

register(
    id='FrankaPlanarPush-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_planar_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_planar_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'object_init_perturb': {'high':[0.01, 0.02, 0.01], 'low':[-0.01, -0.02, 0.0]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.1, 0.48, 0.78], 'low':[0.1, 0.48, 0.78]},
        'init_qpos': [0.5691, 0.697, 0.0563, -1.998, -0.0809, 1.1216, 2.2556, 0.03, 0.03],
        'pos_limit_low': [-0.4, 0.3, 0.8, 3.14, 0, -0.3, 0.025],
        'pos_limit_high': [0.4, 0.8, 0.9, 3.14, 0, 0.3, 0.025],
        'success_mask': {'left':40, 'right': 130, 'top':70, 'bottom':110}
    }
)

register(
    id='FrankaPlanarPushReal-v0',
    entry_point='mj_envs.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_planar_push_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_planar_push_v0.config',
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'object_init_perturb': {'high':[0.01, 0.02, 0.01], 'low':[-0.01, -0.02, 0.0]},
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.1, 0.48, 0.78], 'low':[0.1, 0.48, 0.78]},
        #'init_qpos': [0.0861, 0.6119, 0.6276, -2.0062, -0.4312, 0.8063, 1.778, 0.0, 0.0],
        'init_qpos': [0.467,0.309,0.088,-2.344,-0.145,1.054,1.4,0.0,0.0],
        #'pos_limit_low': [-0.4, 0.2, 0.9, 3.14, 0, -0.79, 0.0],
        #'pos_limit_high': [0.4, 0.7, 0.95, 3.14, 0, -0.69, 0.0],
        'pos_limit_low': [-0.4, 0.2, 0.9, 3.14, 0, -1.57, 0.0],
        'pos_limit_high': [0.4, 0.7, 0.95, 3.14, 0, -0.17, 0.0],
        'success_mask': {'left':50, 'right': 140, 'top':105, 'bottom':145, 'thresh':0.2}
    }
)

# Reach to random target using visual inputs
def register_push_visual_envs(env_name, encoder_type, cams, real=False, real_cams=None):
    obs_keys = ['qp', 'qv', 'grasp_pos', 'object_err', 'target_err']
    for cam in cams:
        obs_keys.append('rgb:'+cam+':224x224:{}'.format(encoder_type))
        obs_keys.append('d:'+cam+':224x224:{}'.format(encoder_type))
    if real:
        for cam in real_cams:
            obs_keys.append('rgb:'+cam+':224x224:{}'.format(encoder_type))
            obs_keys.append('d:'+cam+':224x224:{}'.format(encoder_type))
       
    register_env_variant(
        env_id='{}-v0'.format(env_name),
        variant_id='{}_v{}-v0'.format(env_name, encoder_type),
        variants={'obs_keys': obs_keys
        },
        silent=True
    )
push_cams =  ['top_cam', 'Franka_wrist_cam']
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_push_visual_envs('FrankaBinPush', enc, cams=push_cams)
    register_push_visual_envs('FrankaBinPushReal', enc, cams=push_cams)

for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_push_visual_envs('FrankaHangPush', enc, cams=push_cams)

planar_push_cams = ['right_cam', 'Franka_wrist_cam']
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_push_visual_envs('FrankaPlanarPush', enc, cams=planar_push_cams)
    register_push_visual_envs('FrankaPlanarPushReal', enc, cams=planar_push_cams)
    
# FRANKA PICK =======================================================================
register(
    id='FrankaPickPlaceFixed-v0',
    entry_point='mj_envs.envs.arms.pick_place_v0:PickPlaceV0',
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
    entry_point='mj_envs.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_busbin_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_site_name': "end_effector",
         #'object_site_names': ["obj0","obj1","obj2"],
        'object_site_names': ["obj0"],
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.235, 0.45, 1.185], 'low':[0.235, 0.43, 1.185]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]}
    }
)

register(
    id='FrankaPickPlaceRandomReal-v0',
    entry_point='mj_envs.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_busbin_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_site_name': "end_effector",
        'object_site_names': ["obj0"],
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.235, 0.45, 1.185], 'low':[0.235, 0.43, 1.185]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]}
    }
)

register(
    id='FrankaPickPlaceRandomMultiObj-v0',
    entry_point='mj_envs.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_busbin_multi_obj_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_site_name': "end_effector",
         'object_site_names': ["obj0","obj1","obj2"],
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[0.235, 0.45, 1.185], 'low':[0.235, 0.43, 1.185]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]}
    }
)

# Reach to random target using visual inputs
def register_pick_place_visual_envs(env_name, encoder_type, real=False):
    obs_keys = ['qp', 'qv', 'grasp_pos', 'object_err', 'target_err',
                "rgb:left_cam:224x224:{}".format(encoder_type),
                "d:left_cam:224x224:{}".format(encoder_type),
                "rgb:right_cam:224x224:{}".format(encoder_type),
                "d:right_cam:224x224:{}".format(encoder_type)]
    if real:
        obs_keys.extend(["rgb:top_cam:224x224:{}".format(encoder_type),
                         "d:top_cam:224x224:{}".format(encoder_type),
                         "rgb:Franka_wrist_cam:224x224:{}".format(encoder_type),
                         "d:Franka_wrist_cam:224x224:{}".format(encoder_type)])

    register_env_variant(
        env_id='{}-v0'.format(env_name),
        variant_id='{}_v{}-v0'.format(env_name, encoder_type),
        variants={'obs_keys': obs_keys
        },
        silent=True
    )
for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_pick_place_visual_envs('FrankaPickPlaceRandom', enc)

for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_pick_place_visual_envs('FrankaPickPlaceRandomReal', enc, real=True)

for enc in ["r3m18", "r3m34", "r3m50", "1d", "2d"]:
    register_pick_place_visual_envs('FrankaPickPlaceRandomMultiObj', enc)

# FETCH =======================================================================
from mj_envs.envs.arms.reach_base_v0 import ReachBaseV0

# Reach to fixed target
register(
    id='FetchReachFixed-v0',
    entry_point='mj_envs.envs.arms.reach_base_v0:ReachBaseV0',
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
    entry_point='mj_envs.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/fetch/assets/fetch_reach_v0.xml',
        'config_path': curr_dir+'/fetch/assets/fetch_reach_v0.config',
        'robot_site_name': "grip",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.3, .5, 1.2], 'low':[-.3, .1, .8]}
    }
)
