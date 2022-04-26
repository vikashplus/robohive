""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
from gym.envs.registration import register

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# Appliences ============================================================================
print("RS:> Registering Appliances Envs")

from mj_envs.envs.multi_task.common.franka_appliance_v1 import FrankaAppliance
obs_keys_wt = {"robot_jnt": 1.0, "objs_jnt": 1.0, "obj_goal": 1.0, "end_effector": 1.0, "microhandle_site_err":1}
# MICROWAVE
register(
    id="franka_micro_open-v3",
    entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
    max_episode_steps=75*2,
    kwargs={
        "model_path": CURR_DIR + "/../common/microwave/franka_microwave.xml",
        "config_path": CURR_DIR + "/../common/microwave/franka_microwave.config",
        "obj_init": {"micro0joint": 0},
        "obj_goal": {"micro0joint": -1.25},
        "obj_interaction_site": ("microhandle_site",),
        "obj_jnt_names": ("micro0joint",),
        "interact_site": "microhandle_site",
    },
)

register(
    id="franka_micro_close-v3",
    entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/../common/microwave/franka_microwave.xml",
        "config_path": CURR_DIR + "/../common/microwave/franka_microwave.config",
        "obj_init": {"micro0joint": -1.25},
        "obj_goal": {"micro0joint": 0},
        "obj_interaction_site": ("microhandle_site",),
        "obj_jnt_names": ("micro0joint",),
        "interact_site": "microhandle_site",
    },
)
register(
    id="franka_micro_random-v3",
    entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/../common/microwave/franka_microwave.xml",
        "config_path": CURR_DIR + "/../common/microwave/franka_microwave.config",
        "obj_init": {"micro0joint": (-1.25, 0)},
        "obj_goal": {"micro0joint": (-1.25, 0)},
        "obj_interaction_site": ("microhandle_site",),
        "obj_jnt_names": ("micro0joint",),
        "obj_body_randomize": ("microwave",),
        "interact_site": "microhandle_site",
    },
)

# SLIDE-CABINET
obs_keys_wt = {"robot_jnt": 1.0, "objs_jnt": 1.0, "obj_goal": 1.0, "end_effector": 1.0, "slide_site_err":1}
register(
    id="franka_slide_open-v3",
    entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.xml",
        "config_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.config",
        "obj_init": {"slidedoor_joint": 0},
        "obj_goal": {"slidedoor_joint": .44},
        "obj_interaction_site": ("slide_site",),
        "obj_jnt_names": ("slidedoor_joint",),
        "interact_site": "slide_site",
    },
)
register(
    id="franka_slide_close-v3",
    entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.xml",
        "config_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.config",
        "obj_init": {"slidedoor_joint": .44},
        "obj_goal": {"slidedoor_joint": 0},
        "obj_interaction_site": ("slide_site",),
        "obj_jnt_names": ("slidedoor_joint",),
        "interact_site": "slide_site",
    },
)
register(
    id="franka_slide_random-v3",
    entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.xml",
        "config_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.config",
        "obj_init": {"slidedoor_joint": (0, .44)},
        "obj_goal": {"slidedoor_joint": (0, .44)},
        "obj_interaction_site": ("slide_site",),
        "obj_jnt_names": ("slidedoor_joint",),
        "obj_body_randomize": ("slidecabinet",),
        "interact_site": "slide_site",
    },
)


# Kitchen-V3 ============================================================================
# In this version of the environment, the observations consist of the
# distance between end effector and all relevent objects in the scene

print("RS:> Registering Kitchen Envs")
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

# Kitchen (base-env; obj_init==obj_goal => do nothing in the env)
register(
    id="kitchen-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=280,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
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
        "obs_keys_wt": obs_keys_wt,
    },
)

# Microwave door
register(
    id="kitchen_micro_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"micro0joint": 0},
        "obj_goal": {"micro0joint": -1.25},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "microhandle_site",
    },
)
register(
    id="kitchen_micro_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"micro0joint": -1.25},
        "obj_goal": {"micro0joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "microhandle_site",
    },
)

# Right hinge cabinet
register(
    id="kitchen_rdoor_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"rightdoorhinge": 0},
        "obj_goal": {"rightdoorhinge": 1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "rightdoor_site",
    },
)
register(
    id="kitchen_rdoor_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"rightdoorhinge": 1.57},
        "obj_goal": {"rightdoorhinge": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "rightdoor_site",
    },
)

# Left hinge cabinet
register(
    id="kitchen_ldoor_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"leftdoorhinge": 0},
        "obj_goal": {"leftdoorhinge": -1.25},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "leftdoor_site",
    },
)
register(
    id="kitchen_ldoor_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"leftdoorhinge": -1.25},
        "obj_goal": {"leftdoorhinge": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "leftdoor_site",
    },
)

# Slide cabinet
register(
    id="kitchen_sdoor_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"slidedoor_joint": 0},
        "obj_goal": {"slidedoor_joint": 0.44},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "slide_site",
    },
)
register(
    id="kitchen_sdoor_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"slidedoor_joint": 0.44},
        "obj_goal": {"slidedoor_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "slide_site",
    },
)

# Lights
register(
    id="kitchen_light_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"lightswitch_joint": 0},
        "obj_goal": {"lightswitch_joint": -0.7},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "light_site",
    },
)
register(
    id="kitchen_light_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"lightswitch_joint": -0.7},
        "obj_goal": {"lightswitch_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "light_site",
    },
)

# Knob4
register(
    id="kitchen_knob4_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob4_joint": 0},
        "obj_goal": {"knob4_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob4_site",
    },
)
register(
    id="kitchen_knob4_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob4_joint": -1.57},
        "obj_goal": {"knob4_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob4_site",
    },
)

# Knob3
register(
    id="kitchen_knob3_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob3_joint": 0},
        "obj_goal": {"knob3_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob3_site",
    },
)
register(
    id="kitchen_knob3_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob3_joint": -1.57},
        "obj_goal": {"knob3_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob3_site",
    },
)

# Knob2
register(
    id="kitchen_knob2_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob2_joint": 0},
        "obj_goal": {"knob2_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob2_site",
    },
)
register(
    id="kitchen_knob2_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob2_joint": -1.57},
        "obj_goal": {"knob2_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob2_site",
    },
)

# Knob1
register(
    id="kitchen_knob1_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob1_joint": 0},
        "obj_goal": {"knob1_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob1_site",
    },
)
register(
    id="kitchen_knob1_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob1_joint": -1.57},
        "obj_goal": {"knob1_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob1_site",
    },
)



# (DEPRICATED) Kitchen-V2  ============================================================================
register(
    id="kitchen-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=280,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {},
    },
)

# Open Microwave door
register(
    id="kitchen_micro_open-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"micro0joint": -1.25},
    },
)

# Open right hinge cabinet
register(
    id="kitchen_rdoor_open-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"rightdoorhinge": 1.57},
    },
)

# Open left hinge cabinet
register(
    id="kitchen_ldoor_open-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"leftdoorhinge": -1.25},
    },
)

# Open slide cabinet
register(
    id="kitchen_sdoor_open-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"slidedoor_joint": 0.44},
    },
)

# Lights on (left)
register(
    id="kitchen_light_on-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"lightswitch_joint": -0.7},
    },
)

# Knob4 on
register(
    id="kitchen_knob4_on-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"knob4_joint": -1.57},
    },
)

# Knob3 on
register(
    id="kitchen_knob3_on-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"knob3_joint": -1.57},
    },
)

# Knob2 on
register(
    id="kitchen_knob2_on-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"knob2_joint": -1.57},
    },
)

# Knob1 on
register(
    id="kitchen_knob1_on-v2",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {"knob1_joint": -1.57},
    },
)
