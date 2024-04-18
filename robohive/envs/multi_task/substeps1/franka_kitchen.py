""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
from robohive.utils import gym; register=gym.register

from robohive.envs.multi_task.common.franka_kitchen_v2 import FrankaKitchen
import copy

print("RoboHive:> Registering FrankaKitchen (FK1) Envs")

# ===================================================================
# ENVs are provided in three VARIANTs (controlling env's behavior)
# - Fixed-v4:       State based stationary environments with no randomization
# - Random-v4:      State based environments with random robot initialization (joint pose + relative position wrt to kitchen)
# - Random_v2d-v4:  Visual environment with random robot initialization (joint pose + relative position wrt to kitchen)
# ===================================================================


# ===================================================================
# ENVs are organized into SUBGROUPS:
# - SUBGROUPS are arranged to ensure diversity and a balance between the groups
# - SUBGROUPS are helpful for standardizing partial results, Test-Train spilt, generalization studies, etc
# - Without these subgroups, random subsets were being picked by different papers making comparisons difficult
# ===================================================================

# Fixed-v4: State based stationary environments with no randomization
FK1_FIXED_5A = [
    "FK1_MicroOpenFixed-v4",
    "FK1_Knob1OnFixed-v4",
    "FK1_Knob2OffFixed-v4",
    "FK1_SdoorOpenFixed-v4",
    "FK1_LdoorOpenFixed-v4"]
FK1_FIXED_5B = [
    "FK1_MicroCloseFixed-v4",
    "FK1_Knob1OffFixed-v4",
    "FK1_Knob2OnFixed-v4",
    "FK1_LightOnFixed-v4",
    "FK1_RdoorOpenFixed-v4",]
FK1_FIXED_5C = [
    "FK1_Stove1KettleFixed-v4",
    "FK1_Knob3OnFixed-v4",
    "FK1_Knob4OffFixed-v4",
    "FK1_SdoorCloseFixed-v4",
    "FK1_RdoorCloseFixed-v4"]
FK1_FIXED_5D = [
    "FK1_Stove4KettleFixed-v4",
    "FK1_Knob3OffFixed-v4",
    "FK1_Knob4OnFixed-v4",
    "FK1_LightOffFixed-v4",
    "FK1_LdoorCloseFixed-v4"]
FK1_FIXED_20 = FK1_FIXED_5A+FK1_FIXED_5B+FK1_FIXED_5C+FK1_FIXED_5D

# Random-v4: State based environments with random robot initialization (joint pose + relative position wrt to kitchen)
FK1_RANDOM_5A = [
    "FK1_MicroOpenRandom-v4",
    "FK1_Knob1OnRandom-v4",
    "FK1_Knob2OffRandom-v4",
    "FK1_SdoorOpenRandom-v4",
    "FK1_LdoorOpenRandom-v4"]
FK1_RANDOM_5B = [
    "FK1_MicroCloseRandom-v4",
    "FK1_Knob1OffRandom-v4",
    "FK1_Knob2OnRandom-v4",
    "FK1_LightOnRandom-v4",
    "FK1_RdoorOpenRandom-v4",]
FK1_RANDOM_5C = [
    "FK1_Stove1KettleRandom-v4",
    "FK1_Knob3OnRandom-v4",
    "FK1_Knob4OffRandom-v4",
    "FK1_SdoorCloseRandom-v4",
    "FK1_RdoorCloseRandom-v4"]
FK1_RANDOM_5D = [
    "FK1_Stove4KettleRandom-v4",
    "FK1_Knob3OffRandom-v4",
    "FK1_Knob4OnRandom-v4",
    "FK1_LightOffRandom-v4",
    "FK1_LdoorCloseRandom-v4"]
FK1_RANDOM_20 = FK1_RANDOM_5A+FK1_RANDOM_5B+FK1_RANDOM_5C+FK1_RANDOM_5D

# Random_v2d-v4:  Visual environment with random robot initialization (joint pose + relative position wrt to kitchen)
FK1_RANDOM_V2D_5A = [
    "FK1_MicroOpenRandom_v2d-v4",
    "FK1_Knob1OnRandom_v2d-v4",
    "FK1_Knob2OffRandom_v2d-v4",
    "FK1_SdoorOpenRandom_v2d-v4",
    "FK1_LdoorOpenRandom_v2d-v4"]
FK1_RANDOM_V2D_5B = [
    "FK1_MicroCloseRandom_v2d-v4",
    "FK1_Knob1OffRandom_v2d-v4",
    "FK1_Knob2OnRandom_v2d-v4",
    "FK1_LightOnRandom_v2d-v4",
    "FK1_RdoorOpenRandom_v2d-v4",]
FK1_RANDOM_V2D_5C = [
    "FK1_Stove1KettleRandom_v2d-v4",
    "FK1_Knob3OnRandom_v2d-v4",
    "FK1_Knob4OffRandom_v2d-v4",
    "FK1_SdoorCloseRandom_v2d-v4",
    "FK1_RdoorCloseRandom_v2d-v4"]
FK1_RANDOM_V2D_5D = [
    "FK1_Stove4KettleRandom_v2d-v4",
    "FK1_Knob3OffRandom_v2d-v4",
    "FK1_Knob4OnRandom_v2d-v4",
    "FK1_LightOffRandom_v2d-v4",
    "FK1_LdoorCloseRandom_v2d-v4"]
FK1_RANDOM_V2D_20 = FK1_RANDOM_V2D_5A+FK1_RANDOM_V2D_5B+FK1_RANDOM_V2D_5C+FK1_RANDOM_V2D_5D


# ===================================================================
# Global Configs
# ===================================================================
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = CURR_DIR + "/../common/kitchen/franka_kitchen.xml"
CONFIG_PATH = CURR_DIR + "/../common/kitchen/franka_kitchen.config"
ENTRY_POINT = "robohive.envs.multi_task.common.franka_kitchen_v2:FrankaKitchen"
ENV_VERSION = "-v4"
VISUAL_ENCODER = "v2d"

# ===================================================================
# Register env variants
# ===================================================================
def register_all_env_variants(
        task_id:str,                # task
        task_configs:dict,          # task details
        max_episode_steps:int=50,   # Horizon
        random_configs:dict=None    # task randomization details
        ):

    # update env's details
    task_configs.update({
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH
            })

    # register state based fixed variants, no randomization
    register(
        id = task_id+"Fixed"+ENV_VERSION,
        entry_point=ENTRY_POINT,
        max_episode_steps=max_episode_steps,
        kwargs=copy.deepcopy(task_configs)
    )
    # print("'"+task_id+"Fixed"+ENV_VERSION+"'", end=", ")

    # update env's randomization configs: random robot initialization (joint pose + relative position wrt to kitchen)
    if random_configs == None:
        random_configs = {
        "robot_jnt_reset_noise_scale": 0.05,    # Joint noise scale for reset
        "robot_base_reset_range": {             # reset range for base wrt kitchen
            'high':[.1, .1, .1],
            'low':[-.1, -.1 , -.1]}
            }
    task_configs.update(random_configs)

    # register state based random variants
    register(
        id = task_id+"Random"+ENV_VERSION,
        entry_point=ENTRY_POINT,
        max_episode_steps=max_episode_steps,
        kwargs= copy.deepcopy(task_configs)
    )
    # print("'"+task_id+"Fixed"+ENV_VERSION+"'", end=", ")


    # update env's visual configs: random robot initialization (joint pose + relative position wrt to kitchen)
    task_configs.update({
            "visual_keys": FrankaKitchen.DEFAULT_VISUAL_KEYS,
            # override the obs to avoid accidental leakage of oracle state info while using the visual envs
            # using time as dummy obs. time keys are added twice to avoid unintended singleton expansion errors.
            # Use proprioceptive data if needed - proprio_keys to configure, env.get_proprioception() to access
            "obs_keys_wt": ['time', 'time']
            })

    # register visual random variants
    register(
        id = task_id+"Random"+"_"+VISUAL_ENCODER+ENV_VERSION,
        entry_point=ENTRY_POINT,
        max_episode_steps=max_episode_steps,
        kwargs=copy.deepcopy(task_configs)
    )
    # print(task_id+"Random"+"_"+VISUAL_ENCODER+ENV_VERSION, end=", ")

# ===================================================================
# Define all tasks
# ===================================================================

# Kitchen (base-env; obj_init==obj_goal => do nothing in the env)
register_all_env_variants(
    task_id="FK1_Relax",
    max_episode_steps=280,
    task_configs={
        "obj_goal": {},
        "obj_init": {
            "knob1_joint": 0,
            "knob2_joint": 0,
            "knob3_joint": 0,
            "knob4_joint": 0,
            "lightswitch_joint": 0,
            "slidedoor_joint": 0,
            "micro0joint": 0,
            "rightdoorhinge": 0,
            "leftdoorhinge": 0,
            },
        }
)

# Microwave
register_all_env_variants(
    task_id="FK1_MicroOpen",
    task_configs={
        "obj_init": {"micro0joint": 0},
        "obj_goal": {"micro0joint": -1.25},
        "interact_site": "microhandle_site"
        }
)
register_all_env_variants(
    task_id="FK1_MicroClose",
    task_configs={
        "obj_init": {"micro0joint": -1.25},
        "obj_goal": {"micro0joint": 0},
        "interact_site": "microhandle_site",
    },
)

# Right hinge cabinet
register_all_env_variants(
    task_id="FK1_RdoorOpen",
    task_configs={
        "obj_init": {"rightdoorhinge": 0},
        "obj_goal": {"rightdoorhinge": 1.57},
        "interact_site": "rightdoor_site",
    },
)
register_all_env_variants(
    task_id="FK1_RdoorClose",
    task_configs={
        "obj_init": {"rightdoorhinge": 1.57},
        "obj_goal": {"rightdoorhinge": 0},
        "interact_site": "rightdoor_site",
    },
)

# Left hinge cabinet door open
register_all_env_variants(
    task_id="FK1_LdoorOpen",
    task_configs={
        "obj_init": {"leftdoorhinge": 0},
        "obj_goal": {"leftdoorhinge": -1.25},
        "interact_site": "leftdoor_site",
    },
)

# Left hinge cabinet door close
register_all_env_variants(
    task_id="FK1_LdoorClose",
    task_configs={
        "obj_init": {"leftdoorhinge": -1.25},
        "obj_goal": {"leftdoorhinge": 0},
        "interact_site": "leftdoor_site",
    },
)

# Slide cabinet
register_all_env_variants(
    task_id="FK1_SdoorOpen",
    task_configs={
        "obj_init": {"slidedoor_joint": 0},
        "obj_goal": {"slidedoor_joint": 0.44},
        "interact_site": "slide_site",
    },
)
register_all_env_variants(
    task_id="FK1_SdoorClose",
    task_configs={
        "obj_init": {"slidedoor_joint": 0.44},
        "obj_goal": {"slidedoor_joint": 0},
        "interact_site": "slide_site",
    },
)

# Lights
register_all_env_variants(
    task_id="FK1_LightOn",
    task_configs={
        "obj_init": {"lightswitch_joint": 0},
        "obj_goal": {"lightswitch_joint": -0.7},
        "interact_site": "light_site",
    },
)
register_all_env_variants(
    task_id="FK1_LightOff",
    task_configs={
        "obj_init": {"lightswitch_joint": -0.7},
        "obj_goal": {"lightswitch_joint": 0},
        "interact_site": "light_site",
    },
)

# Knob4
register_all_env_variants(
    task_id="FK1_Knob4On",
    task_configs={
        "obj_init": {"knob4_joint": 0},
        "obj_goal": {"knob4_joint": -1.57},
        "interact_site": "knob4_site",
    },
)
register_all_env_variants(
    task_id="FK1_Knob4Off",
    task_configs={
        "obj_init": {"knob4_joint": -1.57},
        "obj_goal": {"knob4_joint": 0},
        "interact_site": "knob4_site",
    },
)

# Knob3
register_all_env_variants(
    task_id="FK1_Knob3On",
    task_configs={
        "obj_init": {"knob3_joint": 0},
        "obj_goal": {"knob3_joint": -1.57},
        "interact_site": "knob3_site",
    },
)
register_all_env_variants(
    task_id="FK1_Knob3Off",
    task_configs={
        "obj_init": {"knob3_joint": -1.57},
        "obj_goal": {"knob3_joint": 0},
        "interact_site": "knob3_site",
    },
)

# Knob2
register_all_env_variants(
    task_id="FK1_Knob2On",
    task_configs={
        "obj_init": {"knob2_joint": 0},
        "obj_goal": {"knob2_joint": -1.57},
        "interact_site": "knob2_site",
    },
)
register_all_env_variants(
    task_id="FK1_Knob2Off",
    task_configs={
        "obj_init": {"knob2_joint": -1.57},
        "obj_goal": {"knob2_joint": 0},
        "interact_site": "knob2_site",
    },
)

# Knob1
register_all_env_variants(
    task_id="FK1_Knob1On",
    task_configs={
        "obj_init": {"knob1_joint": 0},
        "obj_goal": {"knob1_joint": -1.57},
        "interact_site": "knob1_site",
    },
)
register_all_env_variants(
    task_id="FK1_Knob1Off",
    task_configs={
        "obj_init": {"knob1_joint": -1.57},
        "obj_goal": {"knob1_joint": 0},
        "interact_site": "knob1_site",
    },
)

# Kettle
register_all_env_variants(
    task_id="FK1_Stove1Kettle",
    task_configs={
        "obj_init": {"kettle0:Tx": -.269},
        "obj_goal": {"kettle0:Tx": .22},
        "interact_site": "kettle_site0",
    },
)
register_all_env_variants(
    task_id="FK1_Stove4Kettle",
    task_configs={
        "obj_init": {"kettle0:Ty": .35},
        "obj_goal": {"kettle0:Ty": .78},
        "interact_site": "kettle_site0",
    },
)