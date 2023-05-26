""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
import sys
import collections
import gym
import numpy as np

from robohive.envs import env_base
from robohive.physics.sim_scene import SimScene
from robohive.utils.quat_math import euler2quat, mat2euler, quat2euler, mat2quat
from robohive.utils.xml_utils import reassign_parent
from robohive.utils.inverse_kinematics import qpos_from_site_pose

class BinPickV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'grasp_pos', 'grasp_rot', 'object_err', 'target_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -1.0,
        "target_dist": -2.5,
        "bonus": 10.0,
        "penalty": -50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):


        # Process model to use Robotiq/DManus as end effector
        raw_sim = SimScene.get_sim(model_path)
        raw_xml = raw_sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
        processed_model_path = model_path[:-4]+"_processed.xml"
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

        os.remove(processed_model_path)
        if processed_obsd_model_path and processed_obsd_model_path!=processed_model_path:
            os.remove(processed_obsd_model_path)

    def _setup(self,
               model_path,
               robot_site_name,
               object_site_name,
               target_site_name,
               target_xyz_range,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               randomize=False,
               pos_limits = {'eef_low': [0.368, -0.25, 0.9, -np.pi, 0, -np.pi, 0.0],
                             'eef_high':[0.72, 0.25,  1.3, np.pi, 2*np.pi, 0, 0.835]},
               vel_limits = {'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.1, 0.2, 0.835],
                             'jnt_slow': [0.1, 0.25, 0.1, 0.25, 0.1, 0.1, 0.2, 0.835],
                             'eef': [0.075, 0.075, 0.15, 0.3, 0.3, 0.3, 0.835],
                             'eef_slow': [0.075, 0.075, 0.075, 0.3, 0.3, 0.3, 0.835]},
               obj_pos_limits = {'low': [0.36, -0.22, 0.855],
                                 'high': [0.64, 0.22, 0.855]},
               min_grab_height=0.905,
               max_slow_height=1.075,
               max_ik=3,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_site_name = object_site_name
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.randomize = randomize
        self.pos_limits = pos_limits
        self.vel_limits = vel_limits
        self.obj_pos_limits = obj_pos_limits
        self.min_grab_height = min_grab_height
        self.max_slow_height = max_slow_height
        self.max_ik = max_ik
        self.last_eef_cmd = None

        self.ik_sim = SimScene.get_sim(model_path)
        self.pos_limits['jnt_low'] = self.sim.model.jnt_range[:self.sim.model.nu, 0]
        self.pos_limits['jnt_high'] = self.sim.model.jnt_range[:self.sim.model.nu, 1]

        for key in pos_limits.keys():
            pos_limits[key] = np.array(pos_limits[key])
        for key in vel_limits.keys():
            vel_limits[key] = np.array(vel_limits[key])
        for key in obj_pos_limits.keys():
            obj_pos_limits[key] = np.array(obj_pos_limits[key])

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        self.viewer_setup(distance=1.25, azimuth=-90, elevation=-20)


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['grasp_pos'] = sim.data.site_xpos[self.grasp_sid].copy()
        obs_dict['grasp_rot'] = mat2quat(self.sim.data.site_xmat[self.grasp_sid].reshape(3, 3).transpose())
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
            ('bonus',   (target_dist<.1) + (target_dist<.05)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  target_dist<.075),
            ('solved',  target_dist<.075),
            ('done',    object_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):

        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()

        if self.randomize:
            # target location
            self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])

            # Randomize obj pose
            obj_jid = self.sim.model.joint_name2id(self.object_site_name)
            reset_qpos[obj_jid:obj_jid+3] = self.np_random.uniform(low=self.obj_pos_limits['low'],
                                                                   high=self.obj_pos_limits['high'])
            reset_qpos[obj_jid+3:obj_jid+7] = euler2quat(self.np_random.uniform(low=(0, 0, -np.pi/2), high=(0, 0, np.pi/2)) ) # random quat

            self.sim.forward()

        obs = super().reset(reset_qpos, reset_qvel, **kwargs)

        cur_pos = self.sim.data.site_xpos[self.grasp_sid]
        cur_rot = mat2euler(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3).transpose())
        cur_rot[np.abs(cur_rot-np.array([np.pi,0.0,0.0]))>np.abs(cur_rot+2*np.pi-np.array([np.pi,0.0,0.0]))] += 2*np.pi
        cur_rot[np.abs(cur_rot-np.array([np.pi,0.0,0.0]))>np.abs(cur_rot-2*np.pi-np.array([np.pi,0.0,0.0]))] -= 2*np.pi
        self.last_eef_cmd = np.concatenate([cur_pos,
                                            cur_rot,
                                            [self.sim.data.qpos[7]]])

        return obs

    def get_ik_action(self, eef_pos, eef_quat):
        for i in range(self.max_ik):

            self.ik_sim.data.qpos[:7] = np.random.normal(self.sim.data.qpos[:7], i*0.1)

            self.ik_sim.data.qpos[2] = 0.0
            self.ik_sim.data.qpos[3] = -2.0
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

        if a.shape[0] == self.sim.model.nu:
            action = a

            # Enforce joint velocity limits
            if self.normalize_act:
                action = (0.5*action+0.5)*(self.pos_limits['jnt_high']-self.pos_limits['jnt_low'])+self.pos_limits['jnt_low']
            action = np.clip(action,
                             self.sim.data.qpos[:self.sim.model.nu]-self.vel_limits['jnt'],
                             self.sim.data.qpos[:self.sim.model.nu]+self.vel_limits['jnt'])
            if self.normalize_act:
                action = 2*(((action - self.pos_limits['jnt_low'])/(self.pos_limits['jnt_high']-self.pos_limits['jnt_low']))-0.5)
        else:
            # Un-normalize cmd
            eef_cmd = a
            if self.normalize_act:
                eef_cmd = (0.5*eef_cmd+0.5)*(self.pos_limits['eef_high']-self.pos_limits['eef_low'])+self.pos_limits['eef_low']
            eef_cmd = np.clip(eef_cmd,
                              self.pos_limits['eef_low'],
                              self.pos_limits['eef_high'])

            # Get current position and rotation of eef
            cur_rot = mat2euler(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3).transpose())
            cur_rot[np.abs(cur_rot-eef_cmd[3:6])>np.abs(cur_rot+2*np.pi-eef_cmd[3:6])] += 2*np.pi
            cur_rot[np.abs(cur_rot-eef_cmd[3:6])>np.abs(cur_rot-2*np.pi-eef_cmd[3:6])] -= 2*np.pi
            cur_pose = np.concatenate([self.sim.data.site_xpos[self.grasp_sid],
                                       cur_rot,
                                       [self.sim.data.qpos[7]]])

            # Enforce cartesian velocity limits
            if cur_pose[2] < self.max_slow_height:# and self.robot.is_hardware:
                eef_cmd = np.clip(eef_cmd,
                                  cur_pose-self.vel_limits['eef_slow'],
                                  cur_pose+self.vel_limits['eef_slow'])
            else:
                eef_cmd = np.clip(eef_cmd,
                                  cur_pose-self.vel_limits['eef'],
                                  cur_pose+self.vel_limits['eef'])

            # Exponential moving average to limit jerk
            assert self.last_eef_cmd is not None
            eef_cmd[:6] = 0.25*eef_cmd[:6] + 0.75*self.last_eef_cmd[:6]

            # Prepare for IK, execute last successful command if IK fails
            eef_pos = eef_cmd[:3]
            eef_elr = eef_cmd[3:6]
            eef_quat= euler2quat(eef_elr)

            ik_result = self.get_ik_action(eef_pos, eef_quat)
            ik_success = ik_result.success
            if not ik_success:
                eef_cmd = self.last_eef_cmd
                eef_pos = eef_cmd[:3]
                eef_elr = eef_cmd[3:6]
                eef_quat = euler2quat(eef_elr)
                ik_result = self.get_ik_action(eef_pos, eef_quat)

            action = ik_result.qpos[:self.sim.model.nu]

            # Check that we are not initiating a grasp at too low of height
            if ((eef_cmd[6] > sys.float_info.epsilon) and
                self.sim.data.site_xpos[self.grasp_sid][2] < self.min_grab_height and self.robot.is_hardware):
                print('Cant grasp this low, z = {}'.format(self.sim.data.site_xpos[self.grasp_sid][2]))
                action[7:self.sim.model.nu] = 0.0
            else:
                action[7:self.sim.model.nu] = eef_cmd[6]

            self.last_eef_cmd = eef_cmd

            # Enforce joint velocity limits
            #if self.robot.is_hardware:
            if cur_pose[2] < self.max_slow_height:
                action = np.clip(action,
                                 self.sim.data.qpos[:self.sim.model.nu]-self.vel_limits['jnt_slow'],
                                 self.sim.data.qpos[:self.sim.model.nu]+self.vel_limits['jnt_slow'])
            else:
                action = np.clip(action,
                                 self.sim.data.qpos[:self.sim.model.nu]-self.vel_limits['jnt'],
                                 self.sim.data.qpos[:self.sim.model.nu]+self.vel_limits['jnt'])

            if self.normalize_act:
                action = 2*(((action - self.pos_limits['jnt_low'])/(self.pos_limits['jnt_high']-self.pos_limits['jnt_low']))-0.5)

        return super().step(action, **kwargs)

class BinPickPolicy():
    def __init__(self,
                 env,
                 seed,
                 move_thresh=0.01,
                 begin_descent_thresh=0.05,
                 begin_grasp_thresh=0.08,
                 align_height=1.075):

        self.env = env
        self.seed = seed
        self.yaw = np.random.uniform(low=self.env.pos_limits['jnt_low'][6],
                                     high=self.env.pos_limits['jnt_high'][6])
        self.last_t = 0.0
        self.stage = 0

        self.last_qp = None
        self.move_thresh = move_thresh

        self.begin_descent_thresh = begin_descent_thresh
        self.begin_grasp_thresh = begin_grasp_thresh
        self.align_height = align_height
        self.gripper_close_thresh = 1e-8 if self.env.robot.is_hardware else 0.01

    def is_moving(self, qp):
        assert(self.last_qp is not None and qp is not None)
        return np.linalg.norm(qp - self.last_qp) > self.move_thresh
    def get_action(self, obs):
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        if not self.env.robot.is_hardware:
            self.yaw = -np.pi/4-quat2euler(obs_dict['qp'][0,0,-4:])[-1]
            while self.yaw < self.env.pos_limits['eef_low'][5]:
                self.yaw += 2*np.pi
            while self.yaw > self.env.pos_limits['eef_high'][5]:
                self.yaw -= 2*np.pi

        action = np.concatenate([obs_dict['grasp_pos'][0, 0, :], [np.pi, 0.0, self.yaw], [obs_dict['qp'][0, 0, 7]]])

        if self.last_t > self.env.sim.data.time:
            # Reset
            self.stage = 0
            self.last_qp = None
        elif self.stage == 0: # Wait until aligned xy
            # Advance to next stage?
            if (np.linalg.norm(obs_dict['object_err'][0,0,:2]) < self.begin_descent_thresh and
               (self.last_qp is not None and not self.is_moving(obs_dict['qp'][0,0,:]))):
                self.stage = 1
        elif self.stage == 1:# Wait until close pregrasp
            if (np.linalg.norm(obs_dict['object_err'][0,0,:3]) < self.begin_grasp_thresh or
               (self.last_qp is not None and not self.is_moving(obs_dict['qp'][0,0,:]))):
                self.stage = 2
        elif self.stage == 2: # Wait until pregrasp has stabilized
            # Advance to next stage?
            if self.last_qp is not None and not self.is_moving(obs_dict['qp'][0,0,:]):
                self.stage = 3
        elif self.stage == 3: # Wait for gripper to start closing
            # Advance to next stage?
            if self.last_qp is not None and obs_dict['qp'][0,0,7] > self.last_qp[7]:
                self.stage = 4
        elif self.stage == 4: # Wait for gripper to stop closing
            if self.last_qp is not None and np.abs(self.last_qp[7] - obs_dict['qp'][0,0,7]) < self.gripper_close_thresh:
                self.stage = 5

        self.last_t = self.env.sim.data.time
        self.last_qp = obs_dict['qp'][0,0,:]

        #print('Stage {}, t {}'.format(self.stage, self.last_t))
        if self.stage == 0: # Align in xy
            action[2] = self.align_height
            action[:2] += 1.5*obs_dict['object_err'][0,0,0:2]
            action[6] = 0.0
        elif self.stage == 1 or self.stage == 2: # Move to pregrasp
            action[:2] += 1.5*obs_dict['object_err'][0,0,0:2]
            action[2] += obs_dict['object_err'][0,0,2]
            action[6] = 0.0
        elif self.stage == 3 or self.stage == 4: # Close gripper
            action[:3] += obs_dict['object_err'][0,0,0:3]
            action[6] = 0.835
        elif self.stage == 5: # Move to target pose
            action[:3] += obs_dict['target_err'][0,0,0:3]
            action[6] = 0.835

        action = np.clip(action, self.env.pos_limits['eef_low'], self.env.pos_limits['eef_high'])

        cur_rot = mat2euler(self.env.sim.data.site_xmat[self.env.grasp_sid].reshape(3,3).transpose())
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot+2*np.pi-action[3:6])] += 2*np.pi
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot-2*np.pi-action[3:6])] -= 2*np.pi
        cur_pos = np.concatenate([self.env.sim.data.site_xpos[self.env.grasp_sid],
                                  cur_rot,
                                  [self.env.sim.data.qpos[7]]])
        action = np.clip(action, cur_pos-self.env.vel_limits['eef'], cur_pos+self.env.vel_limits['eef'])

        # Normalize action to be between -1 and 1
        action = 2*(((action - self.env.pos_limits['eef_low']) / (self.env.pos_limits['eef_high'] - self.env.pos_limits['eef_low'])) - 0.5)

        return action, {'evaluation': action}