""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

print("RoboHive:> Registering Multi-Task (9 subtasks) Envs")
from robohive.envs.multi_task.substeps1.franka_kitchen import register_all_env_variants

# ===================================================================
# ENVs are provided in three VARIANTs (controlling env's behavior)
# - Fixed-v4:       State based stationary environments with no randomization
# - Random-v4:      State based environments with random robot initialization (joint pose + relative position wrt to kitchen)
# - Random_v2d-v4:  Visual environment with random robot initialization (joint pose + relative position wrt to kitchen)
# ===================================================================


# Kitchen (close everything)
register_all_env_variants(
    task_id="FK1_OpenAll",
    max_episode_steps=280,
    task_configs={
        "obj_init": {},
        "obj_goal": {
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
    },
)

register_all_env_variants(
    task_id="FK1_CloseAll",
    max_episode_steps=280,
    task_configs={
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
    },
)