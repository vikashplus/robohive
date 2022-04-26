""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from mj_envs.envs import env_base

VIZ = False

class KitchenBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS_AND_WEIGHTS = {
        "robot_jnt": 1.0,
        "objs_jnt": 1.0,
        "obj_goal": 1.0,
        "goal_err": 1.0,
        "approach_err": 1.0,
    }
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "obj_goal": 1.0,
        "bonus": 0.5,
        "pose": 0.0,  # 0.01
        "approach": 0.5,
    }

    def _setup(self,
               robot_jnt_names,
               obj_jnt_names,
               obj_interaction_site,
               obj_goal,
               interact_site="end_effector",
               obj_init=None,
               obs_keys_wt=list(DEFAULT_OBS_KEYS_AND_WEIGHTS.keys()),
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               # different defaults than what is used in env_base and robot
               frame_skip=40,
               obs_range=(-8, 8),
               act_mode="vel",
               robot_name="Franka_kitchen_sim",
               **kwargs,
    ):

        if VIZ:
            from vtils.plotting.srv_dict import srv_dict

            self.dict_plot = srv_dict()

        # configure env-site
        self.grasp_sid = self.sim.model.site_name2id("end_effector")
        self.obj_interaction_site = obj_interaction_site

        # configure env-robot
        self.robot_dofs = []
        self.robot_ranges = []
        for jnt_name in robot_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.robot_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.robot_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.robot_dofs = np.array(self.robot_dofs)
        self.robot_ranges = np.array(self.robot_ranges)
        self.robot_meanpos = np.mean(self.robot_ranges, axis=1)

        # configure env-objs
        self.obj = {}
        obj_dof_adrs = []
        obj_dof_ranges = []
        for goal_adr, jnt_name in enumerate(obj_jnt_names):
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.obj[jnt_name] = {}
            self.obj[jnt_name]["goal_adr"] = goal_adr
            self.obj[jnt_name]["interact_sid"] = self.sim.model.site_name2id(
                obj_interaction_site[goal_adr]
            )
            self.obj[jnt_name]["dof_adr"] = self.sim.model.jnt_dofadr[jnt_id]
            obj_dof_adrs.append(self.sim.model.jnt_dofadr[jnt_id])
            obj_dof_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.obj["dof_adrs"] = np.array(obj_dof_adrs)
        self.obj["dof_ranges"] = np.array(obj_dof_ranges)
        self.obj["dof_ranges"] = (
            self.obj["dof_ranges"][:, 1] - self.obj["dof_ranges"][:, 0]
        )

        # configure env-obj_goal
        if interact_site == "end_effector":
            print(
                "WARNING: Using the default interaction site of end-effector. \
                  If you wish to evaluate on specific tasks, you should set the interaction site correctly."
            )
        self.input_obj_goal = obj_goal
        self.input_obj_init = obj_init
        self.set_obj_goal(obj_goal=self.input_obj_goal, interact_site=interact_site)

        super()._setup(obs_keys=obs_keys_wt,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       act_mode=act_mode,
                       obs_range=obs_range,
                       robot_name=robot_name,
                       **kwargs)


        self.init_qpos[:] = self.sim.model.key_qpos[0].copy()
        if obj_init:
            self.set_obj_init(self.input_obj_init)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["robot_jnt"] = sim.data.qpos[self.robot_dofs].copy()
        obs_dict["objs_jnt"] = sim.data.qpos[self.obj["dof_adrs"]].copy()
        obs_dict["robot_vel"] = sim.data.qvel[self.robot_dofs].copy() * self.dt
        obs_dict["objs_vel"] = sim.data.qvel[self.obj["dof_adrs"]].copy() * self.dt
        obs_dict["obj_goal"] = self.obj_goal.copy()
        obs_dict["goal_err"] = (
            obs_dict["obj_goal"] - obs_dict["objs_jnt"]
        )  # mix of translational and rotational erros
        obs_dict["approach_err"] = (
            self.sim.data.site_xpos[self.interact_sid]
            - self.sim.data.site_xpos[self.grasp_sid]
        )
        obs_dict["pose_err"] = self.robot_meanpos - obs_dict["robot_jnt"]
        obs_dict["end_effector"] = self.sim.data.site_xpos[self.grasp_sid]
        obs_dict["qpos"] = self.sim.data.qpos.copy()
        for site in self.obj_interaction_site:
            site_id = self.sim.model.site_name2id(site)
            obs_dict[site + "_err"] = (
                self.sim.data.site_xpos[site_id]
                - self.sim.data.site_xpos[self.grasp_sid]
            )
        return obs_dict

    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(obs_dict["goal_err"])

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("obj_goal", -np.sum(goal_dist, axis=-1)),
                (
                    "bonus",
                    np.product(goal_dist < 0.75 * self.obj["dof_ranges"], axis=-1)
                    + np.product(goal_dist < 0.25 * self.obj["dof_ranges"], axis=-1),
                ),
                ("pose", -np.sum(np.abs(obs_dict["pose_err"]), axis=-1)),
                ("approach", -np.linalg.norm(obs_dict["approach_err"], axis=-1)),
                # Must keys
                ("sparse", -np.sum(goal_dist, axis=-1)),
                ("solved", np.all(goal_dist < 0.15 * self.obj["dof_ranges"])),
                ("done", False),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        if self.mujoco_render_frames and VIZ:
            self.dict_plot.append(rwd_dict, self.rwd_keys_wt)
            # self.dict_plot.append(rwd_dict)

        return rwd_dict

    # Set object initializations
    # Note: this doesn't directly initializes the objects. One has to either call reset or explicitely use init_qpos manually set the sim.
    def set_obj_init(self, obj_init):
        # resolve goals
        if type(obj_init) is dict:
            # overwrite explicit requests
            for obj_name, obj_val in obj_init.items():
                val = self.np_random.uniform(low=obj_val[0], high=obj_val[1]) if type(obj_val)==tuple else obj_val
                self.init_qpos[self.obj[obj_name]["dof_adr"]] = val
        elif type(obj_init) is np.ndarray:
            assert len(obj_init) == len(
                self.obj["dof_adrs"]
            ), "Check size of provided obj_init"
            self.init_qpos[self.obj["dof_adrs"]] = obj_init.copy()
        else:
            raise TypeError(
                "obj_init must be either a dict<obj_name, obb_init>, or a vector of all obj_init"
            )

    def set_obj_goal(self, obj_goal=None, interact_site=None):

        # resolve goals
        if type(obj_goal) is dict:
            # treat current sim as obj_goal
            self.obj_goal = self.sim.data.qpos[self.obj["dof_adrs"]].copy()
            # overwrite explicit requests
            for obj_name, obj_val in obj_goal.items():
                # import ipdb; ipdb.set_trace()
                val = self.np_random.uniform(low=obj_val[0], high=obj_val[1]) if type(obj_val)==tuple else obj_val
                self.obj_goal[self.obj[obj_name]["goal_adr"]] = val
        elif type(obj_goal) is np.ndarray:
            assert len(obj_goal) == len(self.obj["dof_adrs"]), "Check size of provided obj_goal"
            self.obj_goal = obj_goal
        else:
            raise TypeError(
                "goals must be either a dict<obj_name, obb_goal>, or a vector of all obj_goals"
            )

        # resolve interaction site
        if interact_site is None:  # automatically infer
            goal_err = np.abs(self.sim.data.qpos[self.obj["dof_adrs"]] - self.obj_goal)
            max_goal_err_obj = np.argmax(goal_err)
            for _, obj in self.obj.items():
                if obj["goal_adr"] == max_goal_err_obj:
                    self.interact_sid = obj["interact_sid"]
                    break
        elif type(interact_site) is str:  # overwrite using name
            self.interact_sid = self.sim.model.site_name2id(interact_site)
        elif type(interact_site) is int:  # overwrite using id
            self.interact_sid = interact_site