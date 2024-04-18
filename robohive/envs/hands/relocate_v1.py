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

ADD_BONUS_REWARDS = True
DEFAULT_RWD_KEYS_AND_WEIGHTS = {
    'palm_obj': 1.0,
    'palm_tar':1.0,
    'obj_tar':1.0,
    'bonus':1.0
    }
# DEFAULT_OBS_KEYS = ['hand_jnt', 'palm_obj_err', 'palm_tar_err', 'obj_tar_err'] # DAPG
DEFAULT_OBS_KEYS = ['hand_jnt', 'palm_obj_err', 'palm_tar_err', 'obj_tar_err', 'obj_pos']


class RelocateEnvV1(env_base.MujocoEnv):

    DEFAULT_CREDIT = """\
    DAPG: Demo Augmented Policy Gradient; Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations
        {Aravind Rajeshwaran*, Vikash Kumar*}, Abhiskek Gupta, John Schulman, Emanuel Todorov, and Sergey Levine
        RSS-2018 | https://sites.google.com/view/deeprl-dexterous-manipulation
    """

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(env_credits=self.DEFAULT_CREDIT, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)


    def _setup(self,
            frame_skip=5,
            reward_mode="dense",
            obs_keys=DEFAULT_OBS_KEYS,
            weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs
            ):

        # ids
        sim = self.sim
        self.target_obj_sid = sim.model.site_name2id("target")
        self.S_grasp_sid = sim.model.site_name2id('S_grasp')
        self.obj_bid = sim.model.body_name2id('Object')

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
        self.init_qpos = np.zeros(self.init_qpos.shape)
        self.init_qvel = np.zeros(self.init_qpos.shape)


    def get_rewards_old(self):
        obj_pos  = self.sim.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.sim.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.sim.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 1.0                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                          # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False
        return reward, goal_achieved


    def get_reward_dict(self, obs_dict):
        palm_obj_dist = np.linalg.norm(obs_dict['palm_obj_err'], axis=-1)
        palm_tar_dist = np.linalg.norm(obs_dict['palm_tar_err'], axis=-1)
        obj_tar_dist = np.linalg.norm(obs_dict['obj_tar_err'], axis=-1)
        obj_pos = obs_dict['obj_pos'][:,:,2] if obs_dict['obj_pos'].ndim==3 else obs_dict['obj_pos'][2]
        obj_lifted = obj_pos > 0.04

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('palm_obj', - 0.1 * palm_obj_dist),            # take hand to object
            ('palm_tar', -0.5 * palm_tar_dist * obj_lifted),# make hand go to target
            ('obj_tar', -0.5 * obj_tar_dist * obj_lifted),  # make obj go to target
            ('bonus', 1.0*obj_lifted + 10.0*(obj_tar_dist<0.1) + 20.0*(obj_tar_dist<0.05)),
            # Must keys
            ('sparse',  -1.0*obj_tar_dist),
            ('solved',  obj_tar_dist<0.1),
            ('done',    palm_obj_dist > 0.7),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def get_obs_old(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.sim.data.qpos.ravel()
        obj_pos  = self.sim.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.sim.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.sim.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])


    def get_obs_dict(self, sim):
        # qpos for hand, xpos for obj, xpos for target
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[:-6].copy()
        obs_dict['palm_obj_err'] = sim.data.site_xpos[self.S_grasp_sid] - sim.data.body_xpos[self.obj_bid]
        obs_dict['palm_tar_err'] = sim.data.site_xpos[self.S_grasp_sid] - sim.data.site_xpos[self.target_obj_sid]
        obs_dict['obj_tar_err'] = sim.data.body_xpos[self.obj_bid] - sim.data.site_xpos[self.target_obj_sid]
        # keys missing from DAPG-env but needed for rewards calculations
        obs_dict['obj_pos']  = sim.data.body_xpos[self.obj_bid].copy()
        return obs_dict


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.sim.reset()
        qp = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qv = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(reset_pos=qp, reset_vel=qv, **kwargs)


        self.sim.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.sim.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.sim.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.sim.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.sim.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.sim.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.sim.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.sim.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.sim.set_state(qpos=qp, qvel=qv)
        self.sim.model.body_pos[self.obj_bid] = obj_pos
        self.sim.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()
