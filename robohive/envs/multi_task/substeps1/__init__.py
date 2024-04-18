""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
from robohive.utils import gym; register=gym.register


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# Appliences ============================================================================
print("RoboHive:> Registering Appliances Envs")

from robohive.envs.multi_task.common.franka_appliance_v1 import FrankaAppliance
# MICROWAVE
register(
    id="franka_micro_open-v3",
    entry_point="robohive.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
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
    entry_point="robohive.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
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
    entry_point="robohive.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
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
register(
    id="franka_slide_open-v3",
    entry_point="robohive.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
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
    entry_point="robohive.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
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
    entry_point="robohive.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
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