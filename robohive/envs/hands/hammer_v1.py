""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
from robohive.utils import gym
import numpy as np

from robohive.utils.quat_math import *
from robohive.envs import env_base

# OBS_KEYS = ['hand_jnt', 'obj_vel', 'palm_pos', 'obj_pos', 'obj_rot', 'target_pos', 'nail_impact'] # DAPG
DEFAULT_OBS_KEYS = ['hand_jnt', 'obj_vel', 'palm_pos', 'obj_pos', 'obj_rot', 'target_pos', 'nail_impact', 'tool_pos', 'goal_pos', 'hand_vel']
DEFAULT_RWD_KEYS_AND_WEIGHTS = {
            'palm_obj': 1.0,
            'tool_target': 1.0,
            'target_goal': 1.0,
            'smooth': 1.0,
            'bonus': 1.0
            } # DAPG

class HammerEnvV1(env_base.MujocoEnv):

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
        self.target_obj_sid = sim.model.site_name2id('S_target')
        self.S_grasp_sid = sim.model.site_name2id('S_grasp')
        self.obj_bid = sim.model.body_name2id('object')
        self.tool_sid = sim.model.site_name2id('tool')
        self.goal_sid = sim.model.site_name2id('nail_goal')
        self.target_bid = sim.model.body_name2id('nail_board')
        self.nail_rid = sim.model.sensor_name2id('S_nail')
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


    def get_reward_dict(self, obs_dict):
        # get to hammer
        palm_obj_dist = np.linalg.norm(obs_dict['palm_pos'] - obs_dict['obj_pos'], axis=-1)
        # take hammer head to nail
        tool_target_dist = np.linalg.norm(obs_dict['tool_pos'] - obs_dict['target_pos'], axis=-1)
        # make nail go inside
        target_goal_dist = np.linalg.norm(obs_dict['target_pos'] - obs_dict['goal_pos'], axis=-1)
        # vel magnitude (handled differently in DAPG)
        hand_vel_mag = np.linalg.norm(obs_dict['hand_vel'], axis=-1)
        obj_vel_mag = np.linalg.norm(obs_dict['obj_vel'], axis=-1)
        # lifting tool
        obj_pos = obs_dict['obj_pos'][:,:,2] if obs_dict['obj_pos'].ndim==3 else obs_dict['obj_pos'][2]
        lifted = (obj_pos > 0.04) * (obj_pos > 0.04)

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('palm_obj', - 0.1 * palm_obj_dist),
            ('tool_target', -1.0 * tool_target_dist),
            ('target_goal', -10.0 * target_goal_dist),
            ('smooth', -1e-2 * (hand_vel_mag + obj_vel_mag)),
            ('bonus', 2.0*lifted + 25.0*(target_goal_dist<0.020) + 75.0*(target_goal_dist<0.010)),
            # Must keys
            ('sparse',  -1.0*target_goal_dist),
            ('solved',  target_goal_dist<0.010),
            ('done',    palm_obj_dist > 1.0),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def get_obs_dict(self, sim):
        # qpos for hand, xpos for obj, xpos for target
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[:-6].copy()
        obs_dict['obj_vel'] = np.clip(sim.data.qvel[-6:].copy(), -1.0, 1.0)
        obs_dict['palm_pos'] = sim.data.site_xpos[self.S_grasp_sid].copy()
        obs_dict['obj_pos'] = sim.data.body_xpos[self.obj_bid].copy()
        obs_dict['obj_rot'] = quat2euler(sim.data.body_xquat[self.obj_bid].copy())
        obs_dict['target_pos'] = sim.data.site_xpos[self.target_obj_sid].copy()
        obs_dict['nail_impact'] = np.clip(sim.data.sensordata[self.nail_rid], [-1.0], [1.0])

        # keys missing from DAPG-env but needed for rewards calculations
        obs_dict['tool_pos'] = sim.data.site_xpos[self.tool_sid].copy()
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid].copy()
        obs_dict['hand_vel'] = np.clip(sim.data.qvel[:-6].copy(), -1.0, 1.0)
        return obs_dict


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.sim.reset()
        qp = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qv = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(reset_pos=qp, reset_vel=qv, **kwargs)

        self.sim.model.body_pos[self.target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.sim.data.qpos.ravel().copy()
        qvel = self.sim.data.qvel.ravel().copy()
        board_pos = self.sim.model.body_pos[self.sim.model.body_name2id('nail_board')].copy()
        target_pos = self.sim.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.sim.set_state(qpos=qp, qvel=qv)
        self.sim.model.body_pos[self.sim.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()
