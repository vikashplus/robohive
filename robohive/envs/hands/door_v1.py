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
# NOTES:
#     1. why is qpos[0] not a part of the obs? ==> Hand translation isn't consistent due to randomization. Palm pos is a good substitute

# OBS_KEYS = ['hand_jnt', 'latch_pos', 'door_pos', 'palm_pos', 'handle_pos', 'reach_err', 'door_open'] # DAPG
DEFAULT_OBS_KEYS = ['hand_jnt', 'latch_pos', 'door_pos', 'palm_pos', 'handle_pos', 'reach_err']
# RWD_KEYS = ['reach', 'open', 'smooth', 'bonus'] # DAPG
DEFAULT_RWD_KEYS_AND_WEIGHTS = {'reach':1.0, 'open':1.0, 'bonus':1.0}

class DoorEnvV1(env_base.MujocoEnv):

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
        self.door_hinge_did = sim.model.jnt_dofadr[sim.model.joint_name2id('door_hinge')]
        self.grasp_sid = sim.model.site_name2id('S_grasp')
        self.handle_sid = sim.model.site_name2id('S_handle')
        self.door_bid = sim.model.body_name2id('frame')
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


    def get_obs_dict(self, sim):
        # qpos for hand, xpos for obj, xpos for target
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[1:-2].copy()
        obs_dict['hand_vel'] = sim.data.qvel[:-2].copy()
        obs_dict['handle_pos'] = sim.data.site_xpos[self.handle_sid].copy()
        obs_dict['palm_pos'] = sim.data.site_xpos[self.grasp_sid].copy()
        obs_dict['reach_err'] = obs_dict['palm_pos']-obs_dict['handle_pos']
        obs_dict['door_pos'] = np.array([sim.data.qpos[self.door_hinge_did]])
        obs_dict['latch_pos'] = np.array([sim.data.qpos[-1]])
        obs_dict['door_open'] = 2.0*(obs_dict['door_pos'] > 1.0) -1.0
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(self.obs_dict['reach_err'], axis=-1)
        door_pos = obs_dict['door_pos'][:,:,0] if obs_dict['door_pos'].ndim==3 else obs_dict['door_pos'][0]
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -0.1* reach_dist),
            ('open',    -0.1*(door_pos - 1.57)*(door_pos - 1.57)),
            ('bonus',   2*(door_pos > 0.2) + 8*(door_pos > 1.0) + 10*(door_pos > 1.35)),
            # Must keys
            ('sparse',  door_pos),
            ('solved',  door_pos > 1.35),
            ('done',    reach_dist > 1.0),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.sim.reset()
        qp = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qv = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(reset_pos=qp, reset_vel=qv, **kwargs)

        self.sim.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.sim.model.body_pos[self.door_bid, 1] = self.np_random.uniform(low=0.25, high=0.35)
        self.sim.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        self.sim.forward()
        return self.get_obs()


    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        door_body_pos = self.sim.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)


    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.sim.set_state(qpos=qp, qvel=qv)
        self.sim.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()
