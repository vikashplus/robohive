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
from robohive.utils.quat_math import euler2quat, mat2euler, quat2euler, mat2quat, quat2mat
from robohive.utils.xml_utils import reassign_parent
from robohive.utils.inverse_kinematics import qpos_from_site_pose

class BinReorientV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'grasp_pos', 'grasp_rot', 'object_err', 'target_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -0.0,
        "target_dist": -1.0,
        "bonus": 10.0,
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
               hand_site_name,
               init_qpos = None,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               randomize=False,
               geom_sizes={'high': [0.0585,0.0525], 'low': [0.0585,0.0475]},
               pos_limits = {'eef_low': [0.368, -0.25, 0.86, -np.pi, 0, -np.pi],
                             'eef_high':[0.72, 0.25,  1.3, np.pi, 2*np.pi, np.pi]},
               vel_limits = {'jnt': [0.15, 0.25, 0.1, 0.25, 0.1, 0.1, 0.6], #SET in INIT
                             'jnt_slow': [0.1, 0.25, 0.1, 0.25, 0.1, 0.1, 0.6],
                             'eef': [0.075, 0.075, 0.15, 0.3, 0.3, 0.5],
                             'eef_slow': [0.075, 0.075, 0.1, 0.3, 0.3, 0.5]},
               obj_pos_limits = {'low': [0.475, -0.025, 0.875],
                                 'high': [0.525, 0.025, 0.875]},
               min_grab_height=0.905,
               max_slow_height=1.075, # SET in INIT
               max_ik=3,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_site_name = object_site_name
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.hand_sid = self.sim.model.site_name2id(hand_site_name)
        self.randomize = randomize
        self.geom_sizes = geom_sizes
        self.pos_limits = pos_limits
        self.vel_limits = vel_limits
        self.obj_pos_limits = obj_pos_limits
        self.min_grab_height = min_grab_height
        self.max_slow_height = max_slow_height
        self.max_ik = max_ik
        self.last_eef_cmd = None
        self.random_size = 0.05
        self.ik_sim = SimScene.get_sim(model_path)
        self.pos_limits['jnt_low'] = self.sim.model.jnt_range[:self.sim.model.nu, 0]
        self.pos_limits['jnt_high'] = self.sim.model.jnt_range[:self.sim.model.nu, 1]

        for key in pos_limits.keys():
            pos_limits[key] = np.array(pos_limits[key])
        for key in vel_limits.keys():
            vel_limits[key] = np.array(vel_limits[key])
        for key in obj_pos_limits.keys():
            obj_pos_limits[key] = np.array(obj_pos_limits[key])

        assert(pos_limits['eef_low'].shape == pos_limits['eef_high'].shape)
        if pos_limits['eef_low'].shape[0] == 6:
            pos_limits['eef_low'] = np.concatenate([pos_limits['eef_low'], self.sim.model.jnt_range[:self.sim.model.nu, 0][7:]])
            pos_limits['eef_high'] = np.concatenate([pos_limits['eef_high'], self.sim.model.jnt_range[:self.sim.model.nu, 1][7:]])

        assert(vel_limits['jnt'].shape == vel_limits['jnt_slow'].shape)
        if vel_limits['jnt'].shape[0] == 7:
            vel_limits['jnt'] = np.concatenate([vel_limits['jnt'], 5*np.ones(10)])
            vel_limits['jnt_slow'] = np.concatenate([vel_limits['jnt_slow'], 5*np.ones(10)])

        assert(vel_limits['eef'].shape == vel_limits['eef_slow'].shape)
        if vel_limits['eef'].shape[0] == 6:
            vel_limits['eef'] = np.concatenate([vel_limits['eef'], 5*np.ones(10)])
            vel_limits['eef_slow'] = np.concatenate([vel_limits['eef_slow'], 5*np.ones(10)])

        super()._setup(obs_keys=obs_keys,
                   weighted_reward_keys=weighted_reward_keys,
                   reward_mode=reward_mode,
                   frame_skip=frame_skip,
                   **kwargs)
        if init_qpos is not None:
            self.init_qpos[:len(init_qpos)] = np.array(init_qpos)[:]                       
        self.viewer_setup(distance=1.25, azimuth=-90, elevation=-20)

        if self.pos_limits is not None:
            act_low = -np.ones(self.pos_limits['eef_low'].shape[0]) if self.normalize_act else self.pos_limits['eef_low'].copy()
            act_high = np.ones(self.pos_limits['eef_high'].shape[0]) if self.normalize_act else self.pos_limits['eef_high'].copy()
            self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

    def set_joint_vel(self, jnt_vel, slow_jnt_vel):
        if jnt_vel is not None:
            assert(len(jnt_vel.shape)==1)
            self.vel_limits['jnt'][:jnt_vel.shape[0]] = jnt_vel[:]
            print('Set jnt vel to {}'.format(self.vel_limits['jnt']))

        if slow_jnt_vel is not None:
            assert(len(slow_jnt_vel.shape)==1)
            self.vel_limits['jnt_slow'][:slow_jnt_vel.shape[0]] = slow_jnt_vel[:]
            print('Set slow jnt vel to {}'.format(self.vel_limits['jnt_slow']))

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['grasp_pos'] = sim.data.site_xpos[self.grasp_sid].copy()
        obs_dict['grasp_rot'] = mat2quat(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3).transpose())
        obs_dict['object_err'] = sim.data.site_xpos[self.object_sid]-sim.data.site_xpos[self.hand_sid]
        obs_dict['target_err'] = np.array([np.abs(self.sim.data.site_xmat[self.object_sid][-1] - 1.0)],dtype=np.float)
        obs_dict['object_rot'] = self.sim.data.site_xmat[self.object_sid].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        object_dist = np.linalg.norm(obs_dict['object_err'], axis=-1)
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        upright = 1.0*(self.sim.data.site_xmat[self.object_sid][-1] > 0.999 and
                       (self.sim.data.site_xpos[self.object_sid][:2] >= self.pos_limits['eef_low'][:2]).all() and
                       (self.sim.data.site_xpos[self.object_sid][:2] <= self.pos_limits['eef_high'][:2]).all())
        end_posed = 1.0*(obs_dict['qp'][0,0,8] > 0.5 and obs_dict['qp'] [0,0,8] < 1.0 and
                         obs_dict['qp'] [0,0,12] > 0.75 and obs_dict['qp'] [0,0,12] < 1.25 and
                         obs_dict['qp'] [0,0,15] > 0.5 and obs_dict['qp'] [0,0,15] < 1.0 and
                         obs_dict['grasp_pos'][0,0,2] > 1.05 and obs_dict['grasp_pos'][0,0,2] < 1.1)
        far_th = 1.25

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('object_dist',   object_dist),
            ('target_dist',   target_dist),
            ('bonus',   (target_dist<.01) + (target_dist<.001)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  upright*end_posed),
            ('solved',  upright*end_posed),
            #('done',    object_dist > far_th),
            ('done', False)
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
            reset_qpos[obj_jid+3:obj_jid+7] = euler2quat(self.np_random.uniform(low=(np.pi/2, np.pi/2,0), high=(np.pi/2, 2*np.pi,0)) ) # random quat

            self.random_size = self.np_random.uniform(low=self.geom_sizes['low'],
                                                      high=self.geom_sizes['high'])  # random size
            bid = self.sim.model.body_name2id(self.object_site_name)
            for gid in range(self.sim.model.body_geomnum[bid]):
                gid += self.sim.model.body_geomadr[bid]
                if gid - self.sim.model.body_geomadr[bid] < 2:
                    self.sim.model.geom_size[gid][:2] = self.random_size
                    self.sim.model.geom_pos[gid][2] = self.random_size[-1]
                    if gid - self.sim.model.body_geomadr[bid] == 1:
                        self.sim.model.geom_pos[gid][2] *= -1.0
                    else:
                        self.random_size[0] += 0.5e-4
                elif gid - self.sim.model.body_geomadr[bid] == 2:
                    self.sim.model.geom_size[gid][0] = 0.5*self.sim.model.geom_size[gid-1][0]
                    self.sim.model.geom_pos[gid][2] = 2*self.sim.model.geom_size[gid-2][1]+self.sim.model.geom_size[gid-2][0]-0.5*self.sim.model.geom_size[gid][1]
                elif gid - self.sim.model.body_geomadr[bid] > 2 and gid - self.sim.model.body_geomadr[bid] < 7:
                    self.sim.model.geom_pos[gid][2] = -2*self.random_size[-1]+self.sim.model.geom_size[gid][0]-0.001
                    if gid - self.sim.model.body_geomadr[bid] == 3:
                        self.sim.model.geom_pos[gid][0] = self.random_size[0] - self.sim.model.geom_size[gid][0]-0.001
                    elif gid - self.sim.model.body_geomadr[bid] == 4:
                        self.sim.model.geom_pos[gid][0] = -self.random_size[0] + self.sim.model.geom_size[gid][0] + 0.001
                    elif gid - self.sim.model.body_geomadr[bid] == 5:
                        self.sim.model.geom_pos[gid][1] = self.random_size[0] - self.sim.model.geom_size[gid][0] - 0.001
                    elif gid - self.sim.model.body_geomadr[bid] == 6:
                        self.sim.model.geom_pos[gid][1] = -self.random_size[0] + self.sim.model.geom_size[gid][0] + 0.001
                #self.sim.model.geom_pos[gid] = self.np_random.uniform(low=-1 * self.sim.model.geom_size[gid],
                #                                                      high=self.sim.model.geom_size[gid])  # random pos

                #if (gid - self.sim.model.body_geomadr[bid] > 2 and gid - self.sim.model.body_geomadr[bid] < 7):
                #    self.sim.model.geom_rgba[gid] = self.sim.model.geom_rgba[self.sim.model.body_geomadr[bid]+1]
                #elif (gid - self.sim.model.body_geomadr[bid] > 0 and self.np_random.rand() > 0.5):
                #    self.sim.model.geom_rgba[gid] = self.sim.model.geom_rgba[gid-1]
                #else:
                #    self.sim.model.geom_rgba[gid][:3] = self.np_random.uniform(low=[.2, .2, .2],
                #                                                           high=[.9, .9, .9])  # random color
            self.sim.forward()

        obs = super().reset(reset_qpos, reset_qvel, blocking=False, **kwargs)

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
        assert(a.shape[0] == self.sim.model.nu or a.shape[0] == 16)

        jnt_act_low = self.sim.model.actuator_ctrlrange[:,0].copy()
        jnt_act_high = self.sim.model.actuator_ctrlrange[:,1].copy()
        cur_pose = None

        if a.shape[0] == self.sim.model.nu:
            action = a
            
            if self.normalize_act:
                action = (0.5*action+0.5)*(jnt_act_high-jnt_act_low)+jnt_act_low
        
        else:
            # Un-normalize cmd
            eef_cmd = a
            if self.normalize_act:
                eef_cmd = (0.5*eef_cmd+0.5)*(self.pos_limits['eef_high']-self.pos_limits['eef_low'])+self.pos_limits['eef_low']
            
            '''
            eef_cmd[0] = 0.5
            eef_cmd[1] = 0.0
            eef_cmd[2] = 1.25
            eef_cmd[3] = -1.57
            eef_cmd[4] = 0.0
            eef_cmd[5] = 0.0
            '''

            eef_cmd = np.clip(eef_cmd,
                              self.pos_limits['eef_low'],
                              self.pos_limits['eef_high'])

            # Get current position and rotation of eef
            cur_rot = mat2euler(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3).transpose())
            cur_rot[np.abs(cur_rot-eef_cmd[3:6])>np.abs(cur_rot+2*np.pi-eef_cmd[3:6])] += 2*np.pi
            cur_rot[np.abs(cur_rot-eef_cmd[3:6])>np.abs(cur_rot-2*np.pi-eef_cmd[3:6])] -= 2*np.pi
            cur_pose = np.concatenate([self.sim.data.site_xpos[self.grasp_sid],
                                       cur_rot,
                                       self.sim.data.qpos[7:self.sim.model.nu]])
                
            # Enforce cartesian velocity limits

            if cur_pose[2] < self.max_slow_height and eef_cmd[2] < self.last_eef_cmd[2]:# and self.robot.is_hardware:
                eef_cmd[:3] = np.clip(eef_cmd[:3],
                                      cur_pose[:3]-self.vel_limits['eef_slow'][:3],
                                      cur_pose[:3]+self.vel_limits['eef_slow'][:3])
            else:
                eef_cmd[:3] = np.clip(eef_cmd[:3],
                                      cur_pose[:3]-self.vel_limits['eef'][:3],
                                      cur_pose[:3]+self.vel_limits['eef'][:3])

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

            action = np.zeros(self.sim.model.nu)
            action[:7] = ik_result.qpos[:7]
            action[7:] = eef_cmd[6:]
            
            self.last_eef_cmd = eef_cmd

        # Enforce joint position limits
        action = np.clip(action, jnt_act_low, jnt_act_high)

        # Enforce joint velocity limits
        if cur_pose is not None and cur_pose[2] < self.max_slow_height:
            action = np.clip(action, 
                                self.sim.data.qpos[:self.sim.model.nu]-self.vel_limits['jnt_slow'],
                                self.sim.data.qpos[:self.sim.model.nu]+self.vel_limits['jnt_slow'])
        else:
            action = np.clip(action, 
                                self.sim.data.qpos[:self.sim.model.nu]-self.vel_limits['jnt'],
                                self.sim.data.qpos[:self.sim.model.nu]+self.vel_limits['jnt'])
        
        if self.normalize_act:
            action = 2*(((action - jnt_act_low)/(jnt_act_high-jnt_act_low))-0.5)                     
        

        self.last_ctrl = self.robot.step(ctrl_desired=action,
                                        ctrl_normalized=self.normalize_act,
                                        step_duration=self.dt,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)
        return self.forward(**kwargs)

class TestPolicy():
    def __init__(self,
                 env,
                 seed):
        self.env = env
        self.seed = seed
        self.grasp_pos = None
        self.prev = None

    def get_action(self, obs):
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        if self.grasp_pos is None:
            self.grasp_pos = obs_dict['grasp_pos'][0, 0, :]
            self.grasp_pos[2] = 0.875
        
        #action = np.concatenate([self.grasp_pos, [np.pi, 0.0, 0.0], obs_dict['qp'][0, 0, 7:self.env.sim.model.nu]])
        action = np.concatenate([self.grasp_pos, [np.pi, 0.0, 0.0], [0.0,-0.2,-0.2,-0.2,0.0,-0.2,-0.2,0.0,-0.2,-0.2]])                                    
        action[6:] = np.array([0.57,-0.2,1.5,0.0, # Thumb
                                         0.0,1.0,0.5,     # Middle
                                         0.0,1.0,0.5])    # Pinky   
        print('obs {}, action {}'.format(obs_dict['grasp_pos'][0, 0, :], action[:3]))
        '''
        if self.prev is None:
            self.prev = -2.57
        action[6] = self.prev
        print('act {}, obs {}'.format(action[6], obs_dict['qp'][0,0,7]))
        if obs_dict['qp'][0,0,7] > 0.25:
            #print('Set action -0.75')
            self.prev = -2.57
        elif obs_dict['qp'][0,0,7] < -1.2:
            #print('Set action 0.75')
            self.prev = 0.57
        '''

        action = 2*(((action - self.env.pos_limits['eef_low']) / (np.abs(self.env.pos_limits['eef_high'] - self.env.pos_limits['eef_low'])+1e-8)) - 0.5)
        action = np.clip(action, -1, 1)

        print('pol: {}'.format(action[6]))
        return action, {'evaluation': action}

class BinReorientPolicy():
    def __init__(self,
                 env,
                 seed,
                 move_thresh=0.01,
                 begin_descent_thresh=0.05,
                 begin_grasp_thresh=0.08,
                 align_height=1.1):

        self.env = env
        self.seed = seed
        self.yaw = 0.0
        self.last_t = 0.0
        self.stage = 0

        self.last_qp = None
        self.last_eef = None
        self.move_thresh = move_thresh

        self.begin_descent_thresh = begin_descent_thresh
        self.begin_grasp_thresh = begin_grasp_thresh
        self.align_height = align_height
        self.gripper_close_thresh = 1e-8 if self.env.robot.is_hardware else 0.03
        self.preplace_thresh = 0.05
        self.real_obj_pos = np.array([0.55, 0.0, 0.875]) # 0.86
        self.real_tar_pos = None

        if self.env.robot.is_hardware:
            self.pregrasp_config = np.array([0.57,1.0,1.5,-np.pi/2, # Thumb
                                            0.75,-0.2,-0.2,     # Middle
                                            -0.4,1.0,-0.2])    # Pinky
            self.grasp_config = np.array([0.57,1.5,1.55,-np.pi/2+1.3, # Thumb
                                            0.75,-0.2,-0.2,     # Middle
                                            -0.4,1.55,0.3])    # Pinky
            self.lift_config = np.array([0.57,1.5,1.55,-np.pi/2+1.3, # Thumb
                                            0.75,1.75,-0.2,     # Middle
                                            -0.75,1.55,0.3])    # Pinky
            self.release_config = np.array([0.57,1.5,1.55,-np.pi/2+1.3, # Thumb
                                            0.75,1.2,-0.2,     # Middle
                                            -0.75,1.55,0.3])    # Pinky
        else:
            self.pregrasp_config = np.array([0.2,1.0,1.5,-np.pi/2, # Thumb
                                         0.75,0.0,-0.2,     # Middle
                                         -0.75,1.0,-0.2])    # Pinky
            self.grasp_config = np.array([0.2,1.5,1.5,-np.pi/2, # Thumb
                                            0.75,0.0,-0.2,     # Middle
                                            -0.75,1.5,-0.2])    # Pinky
            self.lift_config = np.array([0.75,1.5,1.6,-np.pi/2+0.45, # Thumb
                                            0.75,0.85,0.5,     # Middle
                                            -0.75,1.75,0.0])    # Pinky
            self.release_config = np.array([0.75,1.5,1.75,-np.pi/2+0.2, # Thumb
                                            0.75,0.35,0.65,     # Middle
                                            -0.75,1.75,0.0])    # Pinky
        self.done_config = np.array([0.57, 0.75, 1.55, -np.pi / 2 + 1.3,  # Thumb
                                        0.75, 1.0, -0.2,  # Middle
                                        -0.75, 0.75, 0.3])  # Pinky
        self.grasp_pose = None
        self.grasp_yaw = None
        self.grasp_count = 0

    def is_moving(self, qp):
        assert(self.last_qp is not None and qp is not None)
        return np.linalg.norm(qp - self.last_qp[:7]) > self.move_thresh

    def set_real_obj_pos(self, real_obj_pos):
        self.real_obj_pos = real_obj_pos

    def set_real_tar_pos(self, real_tar_pos):
        self.real_tar_pos = real_tar_pos

    def set_real_yaw(self, real_yaw):
        self.yaw = real_yaw
        while self.yaw < -np.pi:
            self.yaw += 2*np.pi
        while self.yaw > np.pi:
            self.yaw -= 2*np.pi

    def get_action(self, obs):
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        if not self.env.robot.is_hardware:

            obj_rot_mat = quat2mat(obs_dict['qp'][0, 0, -4:])
            obj_yaw = np.arctan2(obj_rot_mat[0,0],obj_rot_mat[0,2])

            self.yaw = obj_yaw+3*np.pi/4
            while self.yaw < self.env.pos_limits['eef_low'][5]:
                self.yaw += 2*np.pi
            while self.yaw > self.env.pos_limits['eef_high'][5]:
                self.yaw -= 2*np.pi

        cur_rot_mat = quat2mat(obs_dict['grasp_rot'][0,0, :])

        cur_yaw = np.arctan2(cur_rot_mat[0,0], cur_rot_mat[0,1])
        cur_yaw -= np.pi/2.0

        #while np.abs(cur_yaw-self.yaw) > np.abs(cur_yaw-2*np.pi-self.yaw):
        #    cur_yaw -= 2*np.pi
        #while np.abs(cur_yaw-self.yaw) > np.abs(cur_yaw+2*np.pi-self.yaw):
        #    cur_yaw += 2*np.pi

        while cur_yaw < self.env.pos_limits['eef_low'][5]:
            cur_yaw += 2*np.pi
        while cur_yaw > self.env.pos_limits['eef_high'][5]:
            cur_yaw -= 2*np.pi

        action = np.concatenate([obs_dict['grasp_pos'][0, 0, :], [np.pi, 0.0, self.yaw], obs_dict['qp'][0, 0, 7:self.env.sim.model.nu]])

        if self.env.robot.is_hardware and self.real_obj_pos is not None:
            obj_err = self.real_obj_pos-obs_dict['grasp_pos'][0, 0, :]
        else:
            obj_err = obs_dict['object_err'][0,0,:]

        yaw_err = self.yaw - cur_yaw

        if self.last_t > self.env.sim.data.time:
            # Reset
            self.stage = 0
            self.last_qp = None
            self.grasp_pose = None
            self.last_eef = None
            self.grasp_yaw  = None
            self.grasp_count = 0


        elif self.stage == 0: # Wait until aligned xy
            # Advance to next stage?
            #print('Cur obj yaw {}, yaw err {}'.format(self.yaw-3*np.pi/4, yaw_err))
            #print('pos err {}, yaw err {}'.format(np.linalg.norm(obj_err[:2]), np.abs(yaw_err)))
            #print('Yaw err {}'.format(np.abs(yaw_err)))
            if (np.linalg.norm(obj_err[:2]) < self.begin_descent_thresh and
                np.abs(yaw_err) < 0.125
               #(self.last_qp is not None and not self.is_moving(obs_dict['qp'][0,0,:7]))):
            ):
                self.stage = 1
                self.grasp_pose = np.zeros_like(obj_err)
                self.grasp_pose[:2] = obs_dict['grasp_pos'][0, 0, :2].copy() + obj_err[:2].copy()
                if self.env.robot.is_hardware:
                    self.grasp_pose[0] = self.real_obj_pos[0]
                    self.grasp_pose[1] = self.real_obj_pos[1]
                    self.grasp_pose[2] = obs_dict['grasp_pos'][0, 0, 2] + 1.0*obj_err[2]
                    #0.15*np.sin(obs_dict['qp'][0, 0, 6] - 0.57)
                    #0.15*np.cos(obs_dict['qp'][0, 0, 6] - 0.57)
                else:
                    self.grasp_pose[2] = obs_dict['grasp_pos'][0, 0, 2] + 1.2*obj_err[2]
                self.grasp_yaw = cur_yaw
        elif self.stage == 1:
            if self.env.robot.is_hardware:
                if obs_dict['grasp_pos'][0,0,2] < 0.92:
                    self.stage = 2
            else:
                if self.last_eef is not None and np.abs(self.last_eef[2]-obs_dict['grasp_pos'][0,0,2]) < 0.001:
                    self.stage = 3
                    self.grasp_pose[2] += 0.2

        elif self.stage == 2:
            assert(self.env.robot.is_hardware)
            diff = obs_dict['qp'][0,0,7:17].copy()
            diff -= self.grasp_config
            #print('Raw {} {}'.format(obs_dict['qp'][0,0,7:17], self.grasp_config))
            #print('Grasp err {}'.format(np.linalg.norm(diff)))
            if np.linalg.norm(diff) < 0.45:
                self.stage = 3
                self.grasp_pose[2] += 0.19
        elif self.stage == 3:
            if self.env.robot.is_hardware:
                #print(obs_dict['qp'][0,0,12])
                self.grasp_count += 1
                if obs_dict['qp'][0,0,12] > 1.55 and self.grasp_count >= 40:
                    self.stage = 4
                    self.grasp_pose[2] -= 0.04
                    self.grasp_count = 0
            else:
                if obj_rot_mat.flatten()[-1] > 0.95:
                #if obs_dict['grasp_pos'][0,0,2] > self.grasp_pose[2] and obs_dict['qp'][0,0,12] > self.lift_config[5] and obs_dict['qp'][0,0,13] > self.lift_config[6]:
                    self.stage = 4
                    #self.grasp_pose[2] += 0.02
                    self.grasp_pose[2] = 1.1125
        elif self.stage == 4:
            #if self.env.robot.is_hardware:
            self.grasp_count += 1
            if self.grasp_count >= 25:
                self.stage = 5


        #if self.last_qp is not None:
        #    print(np.linalg.norm(self.last_qp[7:self.env.sim.model.nu] - obs_dict['qp'][0, 0, 7:self.env.sim.model.nu]))
        self.last_t = self.env.sim.data.time
        self.last_qp = obs_dict['qp'][0,0,:self.env.sim.model.nu]


        #print('Stage {}, t {}'.format(self.stage, self.last_t))
        if self.stage == 0: # Align in xy
            if self.env.robot.is_hardware:
                action[:2] = self.real_obj_pos[:2] 
            else:
                action[:2] += 1.5*obj_err[0:2]
            action[2] = self.align_height
            action[6:] = self.pregrasp_config
        elif self.stage == 1:
            action[:3] = self.grasp_pose
            action[5] = self.grasp_yaw
            if self.env.robot.is_hardware:
                action[6:] = self.pregrasp_config
            else:
                action[6:] = self.grasp_config
        elif self.stage == 2:
            action[:3] = self.grasp_pose
            action[5] = self.grasp_yaw
            action[6:] = self.grasp_config            
        elif self.stage == 3:
            action[:3] = self.grasp_pose
            action[5] = self.grasp_yaw
            action[6:] = self.lift_config
        elif self.stage == 4:
            action[:3] = self.grasp_pose
            action[5] = self.grasp_yaw
            action[6:] = self.release_config
        elif self.stage == 5:
            action[:3] = self.grasp_pose
            action[5] = self.grasp_yaw
            action[6:] = self.done_config
            #print('height {}'.format(obs_dict['grasp_pos'][0,0,2]))
        self.last_eef = obs_dict['grasp_pos'][0,0,:].copy()
        action = np.clip(action, self.env.pos_limits['eef_low'], self.env.pos_limits['eef_high'])

        cur_rot = mat2euler(self.env.sim.data.site_xmat[self.env.grasp_sid].reshape(3,3).transpose())
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot+2*np.pi-action[3:6])] += 2*np.pi
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot-2*np.pi-action[3:6])] -= 2*np.pi
        cur_pos = np.concatenate([self.env.sim.data.site_xpos[self.env.grasp_sid],
                                  cur_rot,
                                  self.env.sim.data.qpos[7:self.env.sim.model.nu]])
        action[:3] = np.clip(action[:3], cur_pos[:3]-self.env.vel_limits['eef'][:3], cur_pos[:3]+self.env.vel_limits['eef'][:3])
        action[6:] = np.clip(action[6:], cur_pos[6:] - self.env.vel_limits['eef'][6:],cur_pos[6:] + self.env.vel_limits['eef'][6:])

        # Normalize action to be between -1 and 1

        action = 2*(((action - self.env.pos_limits['eef_low']) / (np.abs(self.env.pos_limits['eef_high'] - self.env.pos_limits['eef_low'])+1e-8)) - 0.5)

        action = np.clip(action, -1, 1)
        noise_action = np.clip(action + 0.05*np.random.randn(action.shape[0]),-1,1)

        return noise_action, {'evaluation': action}
