""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
from gym.envs.registration import register

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# Kitchen-V3 ============================================================================
# In this version of the environment, the observations consist of the
# distance between end effector and all relevent objects in the scene

print("RS:> Registering Multi-Task (9 subtasks) Envs")
from mj_envs.envs.multi_task.common.franka_kitchen_v1 import KitchenFrankaFixed, KitchenFrankaRandom, KitchenFrankaDemo

MODEL_PATH = CURR_DIR + "/../common/kitchen/franka_kitchen.xml"
CONFIG_PATH = CURR_DIR + "/../common/kitchen/franka_kitchen.config"

DEMO_ENTRY_POINT = "mj_envs.envs.multi_task.common.franka_kitchen_v1:KitchenFrankaDemo"
RANDOM_ENTRY_POINT = "mj_envs.envs.multi_task.common.franka_kitchen_v1:KitchenFrankaRandom"
FIXED_ENTRY_POINT = "mj_envs.envs.multi_task.common.franka_kitchen_v1:KitchenFrankaFixed"
ENTRY_POINT = RANDOM_ENTRY_POINT

obs_keys_wt = {"robot_jnt": 1.0, "objs_jnt": 1.0, "obj_goal": 1.0, "end_effector": 1.0}
for site in KitchenFrankaFixed.OBJ_INTERACTION_SITES:
    obs_keys_wt[site + "_err"] = 1.0

# Kitchen (close everything)
register(
    id="kitchen_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {},
        "obj_init": {
            "knob1_joint": -1.57,
            "knob2_joint": -1.57,
            "knob3_joint": -1.57,
            "knob4_joint": -1.57,
            "lightswitch_joint": -0.7,
            "slidedoor_joint": 0.44,
            "micro0joint": -1.25,
            "rightdoorhinge": 1.57,
            "leftdoorhinge": -1.25,
        },
        "obs_keys_wt": obs_keys_wt,
    },
)