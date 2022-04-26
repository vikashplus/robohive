""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from mj_envs.envs.multi_task.multi_task_base_v1 import KitchenBase

# ToDo: Get these details from key_frame
DEMO_RESET_QPOS = np.array(
    [
        1.01020992e-01,
        -1.76349747e00,
        1.88974607e00,
        -2.47661710e00,
        3.25189114e-01,
        8.29094410e-01,
        1.62463629e00,
        3.99760380e-02,
        3.99791002e-02,
        2.45778156e-05,
        2.95590127e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.16196258e-05,
        5.08073663e-06,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        -2.68999994e-01,
        3.49999994e-01,
        1.61928391e00,
        6.89039584e-19,
        -2.26122120e-05,
        -8.87580375e-19,
    ]
)
DEMO_RESET_QVEL = np.array(
    [
        -1.24094905e-02,
        3.07730486e-04,
        2.10558046e-02,
        -2.11170651e-02,
        1.28676305e-02,
        2.64535546e-02,
        -7.49515183e-03,
        -1.34369839e-04,
        2.50969693e-04,
        1.06229627e-13,
        7.14243539e-16,
        1.06224762e-13,
        7.19794728e-16,
        1.06224762e-13,
        7.21644648e-16,
        1.06224762e-13,
        7.14243539e-16,
        -1.19464428e-16,
        -1.47079926e-17,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.93530267e-09,
        -1.99505748e-18,
        3.42031125e-14,
        -4.39396125e-17,
        6.64174740e-06,
        3.52969879e-18,
    ]
)

class KitchenFrankaFixed(KitchenBase):

    OBJ_INTERACTION_SITES = (
        "knob1_site",
        "knob2_site",
        "knob3_site",
        "knob4_site",
        "light_site",
        "slide_site",
        "leftdoor_site",
        "rightdoor_site",
        "microhandle_site",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
    )

    OBJ_JNT_NAMES = (
        "knob1_joint",
        "knob2_joint",
        "knob3_joint",
        "knob4_joint",
        "lightswitch_joint",
        "slidedoor_joint",
        "leftdoorhinge",
        "rightdoorhinge",
        "micro0joint",
        "kettle0:Tx",
        "kettle0:Ty",
        "kettle0:Tz",
        "kettle0:Rx",
        "kettle0:Ry",
        "kettle0:Rz",
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

    def _setup(
        self,
        robot_jnt_names=ROBOT_JNT_NAMES,
        obj_jnt_names=OBJ_JNT_NAMES,
        obj_interaction_site=OBJ_INTERACTION_SITES,
        **kwargs,
    ):
        super()._setup(
            robot_jnt_names=robot_jnt_names,
            obj_jnt_names=obj_jnt_names,
            obj_interaction_site=obj_interaction_site,
            **kwargs,
        )


class KitchenFrankaDemo(KitchenFrankaFixed):

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

        super()._setup(**kwargs)

    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qvel = self.init_qvel.copy()
            reset_qpos[self.robot_dofs] = DEMO_RESET_QPOS[self.robot_dofs]
            reset_qvel[self.robot_dofs] = DEMO_RESET_QVEL[self.robot_dofs]
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)


class KitchenFrankaRandom(KitchenFrankaFixed):

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

        super()._setup(**kwargs)

    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)
