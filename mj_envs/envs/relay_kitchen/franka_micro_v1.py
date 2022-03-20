""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np
from mj_envs.utils.quat_math import euler2quat

from mj_envs.envs.relay_kitchen.multi_task_base_v1 import KitchenBase

class FrankaMicroFixed(KitchenBase):

    OBJ_INTERACTION_SITES = (
        "microhandle_site",
    )

    OBJ_JNT_NAMES = (
        "microjoint",
    )

    ROBOT_JNT_NAMES = (
        "panda0_joint1",
        "panda0_joint2",
        "panda0_joint3",
        "panda0_joint4",
        "panda0_joint5",
        "panda0_joint6",
        "panda0_joint7",
        "panda0_finger_joint1",
        "panda0_finger_joint2",
    )

    def __init__(
        self,
        model_path,
        robot_jnt_names=ROBOT_JNT_NAMES,
        obj_jnt_names=OBJ_JNT_NAMES,
        obj_interaction_sites=OBJ_INTERACTION_SITES,
        goal=None,
        interact_site="end_effector",
        obj_init=None,
        **kwargs,
    ):
        KitchenBase.__init__(
            self,
            model_path=model_path,
            robot_jnt_names=robot_jnt_names,
            obj_jnt_names=obj_jnt_names,
            obj_interaction_sites=obj_interaction_sites,
            goal=goal,
            interact_site=interact_site,
            obj_init=obj_init,
            **kwargs,
        )

class FrankaMicroRandom(FrankaMicroFixed):
    def reset(self, reset_qpos=None, reset_qvel=None):

        # import ipdb; ipdb.set_trace()
        bid = self.sim.model.body_name2id('microwave')
        r = self.np_random.uniform(low=.4, high=.7)
        theta = self.np_random.uniform(low=-1.57, high=1.57)
        self.sim.model.body_pos[17][0] = r*np.sin(theta)
        self.sim.model.body_pos[17][1] = 0.5 + r*np.cos(theta)
        self.sim.model.body_quat[17] = euler2quat([0, 0, -theta])
        # import ipdb; ipdb.set_trace()


        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)
