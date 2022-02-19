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

class FrankaApplianceFixed(KitchenBase):

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
        obj_body_names,
        robot_jnt_names=ROBOT_JNT_NAMES,
        **kwargs,
    ):
        self.obj_body_names = obj_body_names
        KitchenBase.__init__(
            self,
            robot_jnt_names=robot_jnt_names,
            **kwargs,
        )

class FrankaApplianceRandom(FrankaApplianceFixed):
    def reset(self, reset_qpos=None, reset_qvel=None):

        for body_name in self.obj_body_names:
            bid = self.sim.model.body_name2id(body_name)
            r = self.np_random.uniform(low=.4, high=.7)
            theta = self.np_random.uniform(low=-1.57, high=1.57)
            self.sim.model.body_pos[bid][0] = r*np.sin(theta)
            self.sim.model.body_pos[bid][1] = 0.5 + r*np.cos(theta)
            self.sim.model.body_quat[bid] = euler2quat([0, 0, -theta])

        # resample init and goal state
        self.set_obj_init(self.input_obj_init)
        self.set_obj_goal(obj_goal=self.input_obj_goal, interact_site=self.interact_sid)

        # random robot pose
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )

        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)
