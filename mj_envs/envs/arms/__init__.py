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
def register_visual_envs(encoder_type):
    register_env_variant(
        env_id='FrankaReachRandom-v0',
        variant_id='FrankaReachRandom_v{}-v0'.format(encoder_type),
        variants={'obs_keys':
                    ['qp', 'qv',
                    "rgb:left_cam:224x224:{}".format(encoder_type),
                    "rgb:right_cam:224x224:{}".format(encoder_type),
                    "rgb:center_cam:224x224:{}".format(encoder_type)]
        },
        silent=True
    )
for enc in ["r3m18", "r3m34", "r3m50", "flat"]:
    register_visual_envs(enc)



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