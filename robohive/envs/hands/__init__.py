""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from robohive.utils import gym; register=gym.register

from robohive.envs.env_variants import register_env_variant
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Hand Envs")

# ==================================================================================
# V1 envs:
#   - env_base class independent of mjrl, making it self contained
#   - Updated obs such that rwd are recoverable from obs
#   - Vectorized obs and rwd calculations
# ==================================================================================

# Swing the door open
register(
    id='door-v1',
    entry_point='robohive.envs.hands:DoorEnvV1',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/DAPG_door.xml',
    }
)
from robohive.envs.hands.door_v1 import DoorEnvV1

# Hammer a nail into the board
register(
    id='hammer-v1',
    entry_point='robohive.envs.hands:HammerEnvV1',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/DAPG_hammer.xml',
    }
)
from robohive.envs.hands.hammer_v1 import HammerEnvV1

# Reposition a pen in hand
register(
    id='pen-v1',
    entry_point='robohive.envs.hands:PenEnvV1',
    max_episode_steps=100,
    kwargs={
        'model_path': curr_dir+'/assets/DAPG_pen.xml',
    }
)
from robohive.envs.hands.pen_v1 import PenEnvV1

# Relocate an object to the target
register(
    id='relocate-v1',
    entry_point='robohive.envs.hands:RelocateEnvV1',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/DAPG_relocate.xml',
    }
)
from robohive.envs.hands.relocate_v1 import RelocateEnvV1

# Reach to random target using visual inputs
def register_visual_envs(env_name, encoder_type):
    register_env_variant(
        env_id='{}-v1'.format(env_name),
        variant_id='{}_v{}-v1'.format(env_name, encoder_type),
        variants={
                # add visual keys to the env
                'visual_keys':[
                    "rgb:vil_camera:224x224:{}".format(encoder_type),
                    "rgb:view_1:224x224:{}".format(encoder_type),
                    "rgb:view_4:224x224:{}".format(encoder_type)],
                # override the obs to avoid accidental leakage of oracle state info while using the visual envs
                # using time as dummy obs. time keys are added twice to avoid unintended singleton expansion errors.
                "obs_keys": ['time', 'time'],
                # add proprioceptive data - proprio_keys to configure, env.get_proprioception() to access
                'proprio_keys':
                    ['hand_jnt'],
        },
        silent=True
    )

for env_name in ["door", "relocate", "hammer", "pen"]:
    for enc in ["r3m18", "r3m34", "r3m50", "rrl18", "rrl34", "rrl50", "2d", "vc1s"]:
        register_visual_envs(env_name, enc)


# ==================================================================================
# Baoding ball
# ==================================================================================
register(
    id='baoding-v1',
    entry_point='robohive.envs.hands:BaodingFixedEnvV1',
    max_episode_steps=200,
    kwargs={
            'model_path': curr_dir+'/assets/PDDM_baoding.xml',
    }
)
from robohive.envs.hands.baoding_v1 import BaodingFixedEnvV1
register(
    id='baoding4th-v1',
    entry_point='robohive.envs.hands:BaodingFixedEnvV1',
    max_episode_steps=200,
    kwargs={
            'model_path': curr_dir+'/assets/PDDM_baoding.xml',
            'n_shifts_per_period':4,
    }
)
register(
    id='baoding8th-v1',
    entry_point='robohive.envs.hands:BaodingFixedEnvV1',
    max_episode_steps=200,
    kwargs={
            'model_path': curr_dir+'/assets/PDDM_baoding.xml',
            'n_shifts_per_period':8,
    }
)
from robohive.envs.hands.baoding_v1 import BaodingFixedEnvV1


# ==================================================================================
# V0 environment
#   - Released as part of the DAPG paper (Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations)
#   - Available hre: https://github.com/vikashplus/robohive/releases/tag/v0.0
# ==================================================================================

# # Swing the door open
# register(
#     id='door-v0',
#     entry_point='robohive.envs.hands:DoorEnvV0',
#     max_episode_steps=200,
# )
# from robohive.envs.hands.door_v0 import DoorEnvV0
#
# # Hammer a nail into the board
# register(
#     id='hammer-v0',
#     entry_point='robohive.envs.hands:HammerEnvV0',
#     max_episode_steps=200,
# )
# from robohive.envs.hands.hammer_v0 import HammerEnvV0
#
# # Reposition a pen in hand
# register(
#     id='pen-v0',
#     entry_point='robohive.envs.hands:PenEnvV0',
#     max_episode_steps=100,
# )
# from robohive.envs.hands.pen_v0 import PenEnvV0
#
# # Relcoate an object to the target
# register(
#     id='relocate-v0',
#     entry_point='robohive.envs.hands:RelocateEnvV0',
#     max_episode_steps=200,
# )
# from robohive.envs.hands.relocate_v0 import RelocateEnvV0
