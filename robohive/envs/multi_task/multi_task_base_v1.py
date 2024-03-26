""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
from robohive.utils import gym
import numpy as np

from robohive.envs import env_base
from robohive.utils.quat_math import mat2euler, quat2euler

VIZ = False

class KitchenBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS_AND_WEIGHTS = {
        "robot_jnt": 1.0,
        "objs_jnt": 1.0,
        "obj_goal": 1.0,
        "goal_err": 1.0,
        "approach_err": 1.0,# task specific
    }

    DEFAULT_VISUAL_KEYS = [
        'rgb:top_cam:256x256:2d',
        'rgb:left_cam:256x256:2d',
        'rgb:right_cam:256x256:2d',
        'rgb:Franka_wrist_cam:256x256:2d']

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "obj_goal": 1.0,
        "bonus": 0.5,
        "pose": 0.0,  # 0.01
        "approach": 0.5,
    }

    DEFAULT_PROPRIO_KEYS_AND_WEIGHTS = {
        "robot_jnt": 1.0,   # radian
        "robot_vel": 1.0,   # radian
        "ee_pose_wrt_robot": 1.0    # [meters, radians]
    }

    def _setup(self,
               robot_jnt_names,
               obj_jnt_names,
               obj_interaction_site,
               obj_goal,
               robot_base_name = "chef",           # Name of the robot chef
               interact_site="end_effector",
               obj_init=None,
               obs_keys_wt=list(DEFAULT_OBS_KEYS_AND_WEIGHTS.keys()),
               proprio_keys_wt=list(DEFAULT_PROPRIO_KEYS_AND_WEIGHTS.keys()),
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
        self.robot_base_bid = self.sim.model.body_name2id(robot_base_name)
        self.robot_base_pos = self.sim.model.body_pos[self.robot_base_bid].copy()

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
        obj_dof_type = []
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
            obj_dof_type.append(self.sim.model.jnt_type[jnt_id]) # record joint type (later used to determine goal_th)
        self.obj["dof_adrs"] = np.array(obj_dof_adrs)
        self.obj["dof_proximity"] = self.get_dof_proximity(obj_dof_ranges, obj_dof_type)

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
                       proprio_keys=proprio_keys_wt,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       act_mode=act_mode,
                       obs_range=obs_range,
                       robot_name=robot_name,
                       **kwargs)


        # Recover init from the saved qposes and input specs
        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        self.init_qvel[:] = self.sim.model.key_qvel[keyFrame_id].copy()
        if obj_init:
            self.set_obj_init(self.input_obj_init)


    def get_dof_proximity(self, obj_dof_ranges, obj_dof_type):
        """
        Get proximity of obj joints based on their joint type and ranges
        """
        small_angular_th = 0.15
        large_angular_th = np.radians(15)
        small_linear_th = 0.15
        large_linear_th = 0.05

        n_dof = len(obj_dof_type)
        dof_prox = np.zeros(n_dof)

        for i_dof in range(n_dof):
            dof_span = obj_dof_ranges[i_dof][1] - obj_dof_ranges[i_dof][0]
            # pick proximity dist based on joint type and scale
            if obj_dof_type[i_dof] == self.sim.lib.mjtJoint.mjJNT_HINGE:
                dof_prox[i_dof] = small_angular_th*dof_span if dof_span<np.pi else large_angular_th
            elif obj_dof_type[i_dof] == self.sim.lib.mjtJoint.mjJNT_SLIDE:
                dof_prox[i_dof] = small_linear_th*dof_span if dof_span<1.0 else large_linear_th
            else:
                raise TypeError("Unsupported Joint Type")
        return dof_prox


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
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

        # End effector global pose
        ee_pos = self.sim.data.site_xpos[self.grasp_sid].copy()
        ee_euler = mat2euler(np.reshape(sim.data.site_xmat[self.grasp_sid],(3,3)))
        obs_dict["ee_pose"] = np.concatenate([ee_pos, ee_euler])

        # End effector local pose wrt robot (for proprioception).
        # ee_pose as proprioception introduced bug in v0.5 that led to information
        # leakage about the relative robot-kitchen positioning
        robot_pos = self.sim.model.body_pos[self.robot_base_bid].copy()
        robot_euler = quat2euler(self.sim.model.body_quat[self.robot_base_bid])
        obs_dict["ee_pose_wrt_robot"] = np.concatenate([ee_pos-robot_pos, ee_euler-robot_euler])

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
                ("bonus",
                    1.0*np.product(goal_dist < 5 * self.obj["dof_proximity"], axis=-1)
                    # np.product(goal_dist < 0.75 * self.obj["dof_ranges"], axis=-1)
                    + 1.0*np.product(goal_dist < 1.67 * self.obj["dof_proximity"], axis=-1),
                    # + np.product(goal_dist < 0.25 * self.obj["dof_ranges"], axis=-1),
                ),
                ("pose", -np.sum(np.abs(obs_dict["pose_err"]), axis=-1)),
                ("approach", -np.linalg.norm(obs_dict["approach_err"], axis=-1)),
                # Must keys
                ("sparse", -np.sum(goal_dist, axis=-1)),
                ("solved", np.all(goal_dist < self.obj["dof_proximity"])),
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