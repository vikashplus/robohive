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
from mj_envs.utils.quat_math import euler2quat
from mj_envs.utils.inverse_kinematics import qpos_from_site_pose
from mujoco_py import load_model_from_path, MjSim
import sys

class PickPlaceV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'grasp_pos', 'object_err', 'target_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -1.0,
        "target_dist": -2.5,
        "bonus": 10.0,
        "penalty": -50,
    }

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


    def _setup(self,
               robot_site_name,
               object_site_names,
               target_site_name,
               target_xyz_range,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               randomize=False,
               geom_sizes={'high':[.05, .05, .05], 'low':[.2, 0.2, 0.2]},
               pos_limit_low=[-0.35, 0.25, 0.76, -3.14, 0, -3.14, 0.0, 0.0],
               pos_limit_high=[0.35, 0.75, 1.5, 3.14, 6.28, 3.14, 0.04, 1.0],
               vel_limit=[0.15, 0.25, 0.1, 0.25, 0.1, 0.25, 0.25, 1.0, 1.0],
               slow_vel_limit=[0.1, 0.25, 0.1, 0.25, 0.1, 0.1, 0.25, 1.0, 1.0],
               eef_vel_limit = [0.15, 0.15, 0.25],
               slow_eef_vel_limit=[0.15, 0.15, 0.075],
               min_grab_height=0.905,
               max_slow_height=1.075,
               max_ik=3,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_site_names = object_site_names
        self.object_sid = self.sim.model.site_name2id(self.object_site_names[0])
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.randomize = randomize
        self.geom_sizes = geom_sizes
        self.pos_limit_low = np.array(pos_limit_low)
        self.pos_limit_high = np.array(pos_limit_high)
        self.vel_limit = vel_limit
        self.slow_vel_limit = slow_vel_limit
        self.eef_vel_limit = eef_vel_limit
        self.slow_eef_vel_limit = slow_eef_vel_limit
        self.min_grab_height = min_grab_height
        self.max_slow_height = max_slow_height
        self.max_ik = max_ik
        self.last_eef_cmd = None 
        self.last_ctrl = None
        self.ik_sim = MjSim(self.sim.model)
        
        self.jnt_low = self.sim.model.jnt_range[:self.sim.model.nu, 0]
        self.jnt_high = self.sim.model.jnt_range[:self.sim.model.nu, 1]

        self.real_obj_pos = np.array([0.0,0.5,1.0])

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        self.action_space = gym.spaces.Box(-np.ones(8), 
                                           np.ones(8), 
                                           dtype=np.float32)

    def set_real_obj_pos(self, real_obj_pos):
        self.real_obj_pos = real_obj_pos

    def set_target_pos(self, target_pos):
        self.sim.model.site_pos[self.target_sid] = target_pos

    def set_init_qpos(self, init_qpos):
        self.init_qpos[:8] = init_qpos[:8]   

    def set_vel_limit(self, vel_limit):
        self.vel_limit = vel_limit

    def set_slow_vel_limit(self, slow_vel_limit):
         self.slow_vel_limit = slow_vel_limit

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['grasp_pos'] = sim.data.site_xpos[self.grasp_sid]
        if self.robot.is_hardware:
            obs_dict['object_err'] = self.real_obj_pos-sim.data.site_xpos[self.grasp_sid]
            obs_dict['target_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.grasp_sid]

        else:
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
            #('bonus',   (object_dist<.1) + (target_dist<.1) + (target_dist<.05)),
            ('bonus',   (target_dist<.1) + (target_dist<.05)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  target_dist<.075),
            ('solved',  target_dist<.075),
            ('done',    object_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, reset_qpos=None, reset_qvel=None):

        if self.randomize:
            # target location
            #self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
            #self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]

            #self.object_sid = self.sim.model.site_name2id(
            #    self.object_site_names[np.random.randint(len(self.object_site_names))])
            # object shapes and locations
            for body in self.object_site_names:
                bid = self.sim.model.body_name2id(body)
            #    self.sim.model.body_pos[bid] += self.np_random.uniform(low=[-.010, -.010, -.010], high=[-.010, -.010, -.010])# random pos
            #    self.sim.model.body_quat[bid] = euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)) ) # random quat

                #for gid in range(self.sim.model.body_geomnum[bid]):
                #    gid+=self.sim.model.body_geomadr[bid]
                #    self.sim.model.geom_type[gid]=self.np_random.randint(low=2, high=7) # random shape
                #    self.sim.model.geom_size[gid]=self.np_random.uniform(low=self.geom_sizes['low'], high=self.geom_sizes['high']) # random size
                #    self.sim.model.geom_pos[gid]=self.np_random.uniform(low=-1*self.sim.model.geom_size[gid], high=self.sim.model.geom_size[gid]) # random pos
                #    self.sim.model.geom_quat[gid]=euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)) ) # random quat
                #    self.sim.model.geom_rgba[gid]=self.np_random.uniform(low=[.2, .2, .2, 1], high=[.9, .9, .9, 1]) # random color
            self.sim.forward()

        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
        
        if self.randomize:
            first_obj_id = self.sim.model.joint_name2id(self.object_site_names[0])
            for body in self.object_site_names:
                obj_jid = self.sim.model.joint_name2id(body)
                base_jid = first_obj_id + 7*(obj_jid-first_obj_id)
                reset_qpos[base_jid:base_jid+3] += self.np_random.uniform(low=[-.01, -.01, -.01], high=[.01, .01, .01])
                reset_qpos[base_jid+3:base_jid+7] = euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)) ) # random quat
        
        if reset_qvel is None:
            reset_qvel = self.init_qvel.copy()
            
        if self.robot.is_hardware:
            self.robot.reset(reset_qpos, reset_qvel, pause_in=False)
            obs = self.get_obs()
        else:
            obs = super().reset(reset_qpos, reset_qvel)
        return obs


    def step(self, a):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        """
        
        ik_success = True
        eef_cmd = None
        invalid_grab = False

        if a.shape[0] == self.sim.model.nu:
            action = a
            if self.robot.is_hardware:
                action = (0.5*action+0.5)*(self.jnt_high-self.jnt_low)+self.jnt_low
                action = np.clip(action, self.sim_obsd.data.qpos[:9]-self.vel_limit, self.sim_obsd.data.qpos[:9]+self.vel_limit)
                action = 2*(((action - self.jnt_low)/(self.jnt_high-self.jnt_low))-0.5)
        else:           
            ik_success = False
            assert(a.flatten().shape[0]==8 or a.flatten().shape[0]==17)
            
            # Un-normalize cmd
            eef_cmd = (0.5*a.flatten()[-8:]+0.5)*(self.pos_limit_high-self.pos_limit_low)+self.pos_limit_low
            eef_cmd = np.clip(eef_cmd, self.pos_limit_low, self.pos_limit_high)
            cur_pos = self.sim_obsd.data.site_xpos[self.grasp_sid]
            if self.robot.is_hardware:
                if cur_pos[2] < self.max_slow_height:
                    eef_cmd[:3] = np.clip(eef_cmd[:3], cur_pos-self.slow_eef_vel_limit, cur_pos+self.eef_vel_limit)
                else:
                    eef_cmd[:3] = np.clip(eef_cmd[:3], cur_pos-self.eef_vel_limit, cur_pos+self.eef_vel_limit)
            eef_pos = eef_cmd[:3]
            eef_elr = eef_cmd[3:6]
            eef_quat= euler2quat(eef_elr)

            for i in range(self.max_ik):

                self.ik_sim.data.qpos[:7] = np.random.normal(self.sim.data.qpos[:7], i*0.1)

                if self.robot.is_hardware:
                    self.ik_sim.data.qpos[2] = 0.0
                    self.ik_sim.forward()

                ik_result = qpos_from_site_pose(physics = self.ik_sim,
                                                site_name = self.sim.model.site_id2name(self.grasp_sid),
                                                target_pos= eef_pos,
                                                target_quat= eef_quat,
                                                inplace=False,
                                                regularization_strength=1.0,
                                                is_hardware=self.robot.is_hardware)
                action = ik_result.qpos[:self.sim.model.nu]
                if ik_result.success:
                    ik_success = True
                    break

            if not ik_success:
                print('IK failed')

            # Check that we are not initiating a grasp at too low of height
            if self.robot.is_hardware and (self.last_eef_cmd is not None and 
                (eef_cmd[6] > self.last_eef_cmd[6] + sys.float_info.epsilon) and 
                self.sim_obsd.data.site_xpos[self.grasp_sid][2] < self.min_grab_height):
                print('Cant grasp this low, z = {}'.format(self.sim_obsd.data.site_xpos[self.grasp_sid][2]))
                invalid_grab = True

            self.last_eef_cmd = eef_cmd
            action[7:9] = eef_cmd[6]

            if self.robot.is_hardware:
                if self.sim_obsd.data.site_xpos[self.grasp_sid][2] < self.max_slow_height:
                    action = np.clip(action, self.sim_obsd.data.qpos[:9]-self.slow_vel_limit, self.sim_obsd.data.qpos[:9]+self.slow_vel_limit)
                else:
                    action = np.clip(action, self.sim_obsd.data.qpos[:9]-self.vel_limit, self.sim_obsd.data.qpos[:9]+self.vel_limit)

            if self.normalize_act:
                action = 2*(((action - self.jnt_low)/(self.jnt_high-self.jnt_low))-0.5)

        if (not self.robot.is_hardware or
            (ik_success and not invalid_grab)):
            self.last_ctrl = self.robot.step(ctrl_desired=action,
                                            ctrl_normalized=self.normalize_act,
                                            step_duration=self.dt,
                                            realTimeSim=self.mujoco_render_frames,
                                            render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()
        done = self.robot.is_hardware and (not ik_success or invalid_grab or (eef_cmd is not None and eef_cmd[7] > 0.5))

        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']) or done, env_info

