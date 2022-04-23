""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from mj_envs.utils.vector_math import calculate_cosine
from mj_envs.utils.quat_math import euler2quat

from mj_envs.envs import env_base

DEFAULT_OBS_KEYS = ['hand_jnt', 'obj_pos', 'obj_vel', 'obj_rot', 'obj_des_rot', 'obj_err_pos', 'obj_err_rot']
DEFAULT_RWD_KEYS_AND_WEIGHTS = {'pos_align':1.0, 'rot_align':1.0, 'drop':1.0, 'bonus':1.0}

class PenEnvV1(env_base.MujocoEnv):

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)


    def _setup(self,
            frame_skip=5,
            reward_mode="dense",
            obs_keys=DEFAULT_OBS_KEYS,
            weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs):

        # ids
        sim = self.sim
        self.target_obj_bid = sim.model.body_name2id("target")
        self.S_grasp_sid = sim.model.site_name2id('S_grasp')
        self.obj_bid = sim.model.body_name2id('Object')
        self.eps_ball_sid = sim.model.site_name2id('eps_ball')
        self.obj_t_sid = sim.model.site_name2id('object_top')
        self.obj_b_sid = sim.model.site_name2id('object_bottom')
        self.tar_t_sid = sim.model.site_name2id('target_top')
        self.tar_b_sid = sim.model.site_name2id('target_bottom')
        self.pen_length = np.linalg.norm(sim.model.site_pos[self.obj_t_sid] - sim.model.site_pos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(sim.model.site_pos[self.tar_t_sid] - sim.model.site_pos[self.tar_b_sid])

        # change actuator sensitivity
        sim.model.actuator_gainprm[sim.model.actuator_name2id('A_WRJ1'):sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        sim.model.actuator_gainprm[sim.model.actuator_name2id('A_FFJ3'):sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        sim.model.actuator_biasprm[sim.model.actuator_name2id('A_WRJ1'):sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        sim.model.actuator_biasprm[sim.model.actuator_name2id('A_FFJ3'):sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        # scales
        self.act_mid = np.mean(sim.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(sim.model.actuator_ctrlrange[:,1]-sim.model.actuator_ctrlrange[:,0])

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)


    def get_obs_dict(self, sim):
        # qpos for hand, xpos for obj, xpos for target
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[:-6].copy()
        obs_dict['obj_pos'] = sim.data.body_xpos[self.obj_bid].copy()
        obs_dict['obj_des_pos'] = sim.data.site_xpos[self.eps_ball_sid].ravel()
        obs_dict['obj_vel'] = sim.data.qvel[-6:].copy()
        obs_dict['obj_rot'] = (sim.data.site_xpos[self.obj_t_sid] - sim.data.site_xpos[self.obj_b_sid])/self.pen_length
        obs_dict['obj_des_rot'] = (sim.data.site_xpos[self.tar_t_sid] - sim.data.site_xpos[self.tar_b_sid])/self.tar_length
        obs_dict['obj_err_pos'] = obs_dict['obj_pos']-obs_dict['obj_des_pos']
        obs_dict['obj_err_rot'] = obs_dict['obj_rot']-obs_dict['obj_des_rot']
        return obs_dict


    def get_reward_dict(self, obs_dict):
        pos_err = obs_dict['obj_err_pos']
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
        # dropped = obs_dict['obj_pos'][:,:,2] < 0.075
        obj_pos = obs_dict['obj_pos'][:,:,2] if obs_dict['obj_pos'].ndim==3 else obs_dict['obj_pos'][2]
        dropped = obj_pos < 0.075

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pos_align',   -1.0*pos_align),
            ('rot_align',   1.0*rot_align),
            ('drop',        -5.0*dropped),
            ('bonus',       10.0*(rot_align > 0.9)*(pos_align<0.075) + 50.0*(rot_align > 0.95)*(pos_align<0.075) ),
            # Must keys
            ('sparse',      -1.0*pos_align+rot_align),
            ('solved',      (rot_align > 0.95)*(~dropped)),
            ('done',        dropped),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        return self.get_obs()


    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)


    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()