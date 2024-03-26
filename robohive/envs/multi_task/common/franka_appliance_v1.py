""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
from robohive.utils import gym
import numpy as np
from robohive.utils.quat_math import euler2quat

from robohive.envs.multi_task.multi_task_base_v1 import KitchenBase

class FrankaAppliance(KitchenBase):

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

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)

    def _setup(
        self,
        obj_body_randomize=None,
        robot_jnt_names=ROBOT_JNT_NAMES,
        **kwargs,
    ):
        self.obj_body_randomize = obj_body_randomize
        super()._setup(
            robot_jnt_names=robot_jnt_names,
            **kwargs,
        )

    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        # randomize object bodies, if requested
        if self.obj_body_randomize:
            for body_name in self.obj_body_randomize:
                bid = self.sim.model.body_name2id(body_name)
                r = self.np_random.uniform(low=.4, high=.7)
                theta = self.np_random.uniform(low=-1.57, high=1.57)
                self.sim.model.body_pos[bid][0] = r*np.sin(theta)
                self.sim.model.body_pos[bid][1] = 0.5 + r*np.cos(theta)
                self.sim.model.body_quat[bid] = euler2quat([0, 0, -theta])

        # resample init and goal state
        self.set_obj_init(self.input_obj_init)
        self.set_obj_goal(obj_goal=self.input_obj_goal, interact_site=self.interact_sid)

        # Noisy robot reset
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )

        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs)
