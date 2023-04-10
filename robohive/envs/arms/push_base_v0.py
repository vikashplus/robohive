""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
import collections
import gym
import numpy as np

from robohive.envs import env_base
from robohive.physics.sim_scene import SimScene
from robohive.utils.xml_utils import reassign_parent
from robohive.utils.quat_math import mat2euler, euler2quat, mat2quat
from robohive.utils.inverse_kinematics import qpos_from_site_pose

class PushBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'grasp_pos', 'grasp_rot', 'object_err', 'target_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -1.0,
        "target_dist": -5.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # Process model to use Robotiq/DManus as end effector
        raw_sim = SimScene.get_sim(model_path)
        raw_xml = raw_sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
        processed_model_path = model_path[:-4]+"_processed.xml"

        cur_xml = None
        if os.path.exists(processed_model_path):
            with open(processed_model_path,'r') as file:
                cur_xml = file.read()

        if cur_xml != processed_xml:
            with open(processed_model_path, 'w') as file:
                file.write(processed_xml)                       

        # Process model to use DManus as end effector
        if obsd_model_path == model_path:
            processed_obsd_model_path = processed_model_path
        elif obsd_model_path:
            raw_sim = SimScene.get_sim(obsd_model_path)
            raw_xml = raw_sim.model.get_xml()
            processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
            processed_obsd_model_path = obsd_model_path[:-4]+"_processed.xml"

            cur_xml = None
            if os.path.exists(processed_obsd_model_path):
                with open(processed_obsd_model_path,'r') as file:
                    cur_xml = file.read()
            if cur_xml != processed_xml:
                with open(processed_obsd_model_path, 'w') as file:
                    file.write(processed_xml)
        else:
            processed_obsd_model_path = None

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
        super().__init__(model_path=processed_model_path, obsd_model_path=processed_obsd_model_path, seed=seed)

        self._setup(processed_model_path, **kwargs)


    def _setup(self,
               model_path,
               robot_site_name,
               object_site_name,
               target_site_name,
               target_xyz_range,
               init_qpos=None,
               pos_limits=None,
               vel_limits=None,
               obj_pos_limits=None,
               max_ik=3,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_site_name = object_site_name
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range

        self.pos_limits = pos_limits
        self.vel_limits = vel_limits
        self.obj_pos_limits = obj_pos_limits
        self.max_ik = max_ik

        self.ik_sim = SimScene.get_sim(model_path)
        self.last_eef_cmd = None

        assert(self.pos_limits is None or
               ('eef_high' in self.pos_limits and 'eef_low' in self.pos_limits ))
        assert(self.vel_limits is None or
               ('jnt' in self.vel_limits and 'eef' in self.vel_limits))
        assert(self.obj_pos_limits is None or
               ('low' in self.obj_pos_limits and 'high' in self.obj_pos_limits))

        if pos_limits is not None:
            for key in pos_limits.keys():
                pos_limits[key] = np.array(pos_limits[key])
        if vel_limits is not None:
            for key in vel_limits.keys():
                vel_limits[key] = np.array(vel_limits[key])
        if obj_pos_limits is not None:
            for key in obj_pos_limits.keys():
                obj_pos_limits[key] = np.array(obj_pos_limits[key])

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        if init_qpos is not None:
            self.init_qpos[:len(init_qpos)] = np.array(init_qpos)[:]

        if self.pos_limits is not None:
            act_low = -np.ones(self.pos_limits['eef_low'].shape[0]) if self.normalize_act else self.pos_limits['eef_low'].copy()
            act_high = np.ones(self.pos_limits['eef_high'].shape[0]) if self.normalize_act else self.pos_limits['eef_high'].copy()
            self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['grasp_pos'] = sim.data.site_xpos[self.grasp_sid].copy()
        obs_dict['grasp_rot'] = mat2quat(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3).transpose())
        obs_dict['object_err'] = sim.data.site_xpos[self.object_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['target_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.object_sid]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        object_dist = np.linalg.norm(obs_dict['object_err'], axis=-1)
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        far_th = 1.25

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('object_dist',   object_dist),
            ('target_dist',   target_dist),
            ('bonus',   (object_dist<.1) + (target_dist<.1) + (target_dist<.05)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*(target_dist<.050)),
            ('solved',  target_dist<.050),
            ('done',    object_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]

        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
        if self.obj_pos_limits is not None:
            obj_jid = self.sim.model.joint_name2id(self.object_site_name)
            reset_qpos[obj_jid:obj_jid+3] = self.np_random.uniform(low=self.obj_pos_limits['low'], high=self.obj_pos_limits['high'])

        obs = super().reset(reset_qpos, reset_qvel, blocking=False, **kwargs)

        cur_pos = self.sim.data.site_xpos[self.grasp_sid]
        cur_rot = mat2euler(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3).transpose())
        if self.pos_limits is not None:
            cur_rot[cur_rot < self.pos_limits['eef_low'][3:6]] += 2*np.pi
            cur_rot[cur_rot > self.pos_limits['eef_high'][3:6]] -= 2 * np.pi

        self.last_eef_cmd = np.concatenate([cur_pos,
                                            cur_rot,
                                            [self.sim.data.qpos[7]]])

        return obs

    def get_ik_action(self, eef_pos, eef_quat):
        for i in range(self.max_ik):

            self.ik_sim.data.qpos[:7] = np.random.normal(self.sim.data.qpos[:7], i*0.1)

            self.ik_sim.data.qpos[2] = 0.0
            self.ik_sim.forward()

            ik_result = qpos_from_site_pose(physics = self.ik_sim,
                                            site_name = self.sim.model.site_id2name(self.grasp_sid),
                                            target_pos= eef_pos,
                                            target_quat= eef_quat,
                                            inplace=False,
                                            regularization_strength=1.0)

            if ik_result.success:
                break
        return ik_result

    def step(self, a, **kwargs):
        assert(len(a.shape) == 1)
        assert(a.shape[0] == self.sim.model.nu or a.shape[0] == 7)

        jnt_act_low = self.sim.model.actuator_ctrlrange[:,0].copy()
        jnt_act_high = self.sim.model.actuator_ctrlrange[:,1].copy()

        if a.shape[0] == self.sim.model.nu:
            action = a
            if self.normalize_act:
                action = (0.5 * action + 0.5) * (jnt_act_high - jnt_act_low) + jnt_act_low

        else:
            # Un-normalize cmd
            eef_cmd = a
            if self.normalize_act:
                eef_cmd = (0.5 * eef_cmd + 0.5) * (self.pos_limits['eef_high'] - self.pos_limits['eef_low']) + \
                          self.pos_limits['eef_low']

            if self.pos_limits is not None:
                eef_cmd = np.clip(eef_cmd,
                                  self.pos_limits['eef_low'],
                                  self.pos_limits['eef_high'])

            # Get current position and rotation of eef
            cur_pose = self.sim.data.site_xpos[self.grasp_sid]

            # Enforce cartesian velocity limits
            if self.vel_limits is not None:
                eef_cmd[:3] = np.clip(eef_cmd[:3],
                                      cur_pose - self.vel_limits['eef'],
                                      cur_pose + self.vel_limits['eef'])

            # Exponential moving average to limit jerk
            assert self.last_eef_cmd is not None
            eef_cmd[:3] = 0.25 * eef_cmd[:3] + 0.75 * self.last_eef_cmd[:3]

            # Prepare for IK, execute last successful command if IK fails
            eef_pos = eef_cmd[:3]
            eef_elr = eef_cmd[3:6]
            eef_quat = euler2quat(eef_elr)

            ik_result = self.get_ik_action(eef_pos, eef_quat)
            ik_success = ik_result.success
            if not ik_success:
                eef_cmd = self.last_eef_cmd
                eef_pos = eef_cmd[:3]
                eef_elr = eef_cmd[3:6]
                eef_quat = euler2quat(eef_elr)
                ik_result = self.get_ik_action(eef_pos, eef_quat)

            action = ik_result.qpos[:self.sim.model.nu]
            action[7:self.sim.model.nu] = eef_cmd[6]

            self.last_eef_cmd = eef_cmd


        # Enforce joint position limits
        action = np.clip(action, jnt_act_low, jnt_act_high)

        # Enforce joint velocity limits
        if self.vel_limits is not None:
            action = np.clip(action,
                                self.sim.data.qpos[:self.sim.model.nu] - self.vel_limits['jnt'],
                                self.sim.data.qpos[:self.sim.model.nu] + self.vel_limits['jnt'])

        if self.normalize_act:
            action = 2 * (((action - jnt_act_low) / (jnt_act_high - jnt_act_low)) - 0.5)

        self.last_ctrl = self.robot.step(ctrl_desired=action,
                                        ctrl_normalized=self.normalize_act,
                                        step_duration=self.dt,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)
        return self.forward(**kwargs)

