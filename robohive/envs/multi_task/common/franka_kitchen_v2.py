""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from robohive.utils import gym
from robohive.envs.multi_task.multi_task_base_v1 import KitchenBase

class FrankaKitchen(KitchenBase):

    ENV_CREDIT = """\
    Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning
        Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman
        CoRL-2019 | https://relay-policy-learning.github.io/
    """

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

    DEFAULT_OBS_KEYS_N_WT = {
        "robot_jnt":1.0,
        "objs_jnt":1.0,
        "obj_goal":1.0,
        "ee_pose":1.0}
    for obj_site in OBJ_INTERACTION_SITES: DEFAULT_OBS_KEYS_N_WT[obj_site+"_err"]=1.0

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
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.ENV_CREDIT)

        self._setup(**kwargs)


    def _setup(
        self,
        obs_keys_wt = DEFAULT_OBS_KEYS_N_WT,
        robot_jnt_reset_noise_scale = 0.0,  # Joint noise amp (around qpos_init) for reset
        robot_base_reset_range = None,      # Randomization range (around the default pose) for the chef's base
        robot_jnt_names=ROBOT_JNT_NAMES,    # Robot joint names
        obj_jnt_names=OBJ_JNT_NAMES,        # Object joint name
        obj_interaction_site=OBJ_INTERACTION_SITES, # Interaction point for object of interest
        **kwargs):

        # configure resets
        self.robot_jnt_reset_noise_scale = robot_jnt_reset_noise_scale
        self.robot_base_range = robot_base_reset_range

        # Setup env
        super()._setup(
            obs_keys_wt=obs_keys_wt,
            robot_jnt_names=robot_jnt_names,
            obj_jnt_names=obj_jnt_names,
            obj_interaction_site=obj_interaction_site,
            **kwargs,
        )


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()

            # add noise to the reset if requested
            reset_qpos[self.robot_dofs] += (
                self.robot_jnt_reset_noise_scale
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )

        # reset franka wrt kitchen
        if self.robot_base_range:
            self.sim.model.body_pos[self.robot_base_bid] = self.robot_base_pos + self.np_random.uniform(**self.robot_base_range)

        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs)