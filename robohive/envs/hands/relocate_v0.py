""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os
from robohive.envs.obs_vec_dict import ObsVecDict
import collections

ADD_BONUS_REWARDS = True
RWD_KEYS = ['palm_obj', 'palm_tar', 'obj_tar', 'bonus']
# OBS_KEYS = ['hand_jnt', 'palm_obj_err', 'palm_tar_err', 'obj_tar_err'] # DAPG
OBS_KEYS = ['hand_jnt', 'palm_obj_err', 'palm_tar_err', 'obj_tar_err', 'obj_pos']
RWD_MODE = 'dense' # dense/ sparse


class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, ObsVecDict):

    DEFAULT_CREDIT = """\
    DAPG: Demo Augmented Policy Gradient; Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations
        {Aravind Rajeshwaran*, Vikash Kumar*}, Abhiskek Gupta, John Schulman, Emanuel Todorov, and Sergey Levine
        RSS-2018 | https://sites.google.com/view/deeprl-dexterous-manipulation
    """

    def __init__(self, *args, **kwargs):

        # get sim
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        sim = mujoco_env.get_sim(model_path=curr_dir+'/assets/DAPG_relocate.xml')
        # ids
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

        # get env
        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)
        self.obs_dict = {}
        self.rwd_dict = {}
        mujoco_env.MujocoEnv.__init__(self, sim=sim, frame_skip=5)

        self.action_space.high = np.ones_like(sim.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(sim.model.actuator_ctrlrange[:,0])

    # step the simulation forward
    def step(self, a):
        # apply action and step
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a*self.act_rng
        self.do_simulation(a, self.frame_skip)

        # observation and rewards
        obs = self.get_obs()
        # obs_old = self.get_obs_old()
        # print(obs_new-obs_old)
        # obs = obs_old

        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # r_old, d_old= self.get_rewards_old()
        # print(r_old - env_info['rwd_'+RWD_MODE], d_old, bool(env_info['solved']))

        return obs, env_info['rwd_'+RWD_MODE], bool(env_info['done']), env_info


    def get_rewards_old(self):
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

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
        obj_lifted = obs_dict['obj_pos'][:,:,2] > 0.04

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
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in RWD_KEYS], axis=0)
        return rwd_dict

    def get_obs_old(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
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

    def get_obs(self):
        """
        Get observations from the environemnt.
        """
        # get obs_dict using the observed information
        self.obs_dict = self.get_obs_dict(self.sim)

        # recoved observation vector from the obs_dict
        t, obs = self.obsdict2obsvec(self.obs_dict, OBS_KEYS)
        return obs

    # use latest obs, rwds to get all info (be careful, information belongs to different timestamps)
    # Its getting called twice. Once in step and sampler calls it as well
    def get_env_infos(self):
        env_info = {
            'time': self.obs_dict['time'][()],
            'rwd_dense': self.rwd_dict['dense'][()],
            'rwd_sparse': self.rwd_dict['sparse'][()],
            'solved': self.rwd_dict['solved'][()],
            'done': self.rwd_dict['done'][()],
            'obs_dict': self.obs_dict,
            'rwd_dict': self.rwd_dict,
        }
        return env_info

    # compute vectorized rewards for paths
    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs_dict = self.obsvec2obsdict(paths["observations"])
        rwd_dict = self.get_reward_dict(obs_dict)

        rewards = rwd_dict[RWD_MODE]
        done = rwd_dict['done']
        # time align rewards. last step is redundant
        done[...,:-1] = done[...,1:]
        rewards[...,:-1] = rewards[...,1:]
        paths["done"] = done if done.shape[0] > 1 else done.ravel()
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths

    def truncate_paths(self, paths):
        hor = paths[0]['rewards'].shape[0]
        for path in paths:
            if path['done'][-1] == False:
                path['terminated'] = False
                terminated_idx = hor
            elif path['done'][0] == False:
                terminated_idx = sum(~path['done'])+1
                for key in path.keys():
                    path[key] = path[key][:terminated_idx+1, ...]
                path['terminated'] = True
        return paths

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
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
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    # evaluate paths and log metrics to logger
    def evaluate_success(self, paths, logger=None):
        num_success = 0
        num_paths = len(paths)
        horizon = self.spec.max_episode_steps # paths could have early termination

        # success if pen within 15 degrees of target for 5 steps
        for path in paths:
            if np.sum(path['env_infos']['solved']) > 5:
                num_success += 1
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
            rwd_dense = np.mean([np.sum(p['env_infos']['rwd_sparse'])/horizon for p in paths]) # return rwd/step
            logger.log_kv('rwd_sparse', rwd_sparse)
            logger.log_kv('rwd_dense', rwd_dense)
            logger.log_kv('success_percentage', success_percentage)

        return success_percentage
