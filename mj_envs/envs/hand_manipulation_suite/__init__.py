""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gym.envs.registration import register
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RS:> Registering Hand Envs")
# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=100,
)
from mj_envs.envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=50,
)
from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

# V0: Old Baoding ball
# register(
#     id='baoding-v0',
#     entry_point='mj_envs.envs.hand_manipulation_suite:BaodingEnvV0',
#     max_episode_steps=200,
# )
# from mj_envs.envs.hand_manipulation_suite.baoding_v0 import BaodingEnvV0

# V0: baoding balls new
register(
    id='baoding-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:BaodingFixedEnvV1',
    max_episode_steps=200,
     kwargs={
            'model_path': curr_dir+'/assets/baoding_v1.mjb',
     }
)
from mj_envs.envs.hand_manipulation_suite.baoding_v1 import BaodingFixedEnvV1
register(
    id='baoding4th-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:BaodingFixedEnvV1',
    max_episode_steps=200,
     kwargs={
            'model_path': curr_dir+'/assets/baoding_v1.mjb',
            'n_shifts_per_period':4,
     }
)
register(
    id='baoding8th-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:BaodingFixedEnvV1',
    max_episode_steps=200,
     kwargs={
            'model_path': curr_dir+'/assets/baoding_v1.mjb',
            'n_shifts_per_period':8,
     }
)
from mj_envs.envs.hand_manipulation_suite.baoding_v1 import BaodingFixedEnvV1
