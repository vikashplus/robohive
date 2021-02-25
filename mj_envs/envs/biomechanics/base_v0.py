import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer, MjSim, load_model_from_path
import os
from mj_envs.utils.obj_vec_dict import ObsVecDict
import collections

RWD_MODE = 'dense' # dense/ sparse

class FingerBaseV0(mujoco_env.MujocoEnv, utils.EzPickle, ObsVecDict):

    def __init__(self,
                obs_keys:list,
                rwd_keys:list,
                sites:tuple,    # fingertip sites of interests
                sim = None,
                model_path:str = None, # only if sim is not provided
                normalize_act = True,
                **kwargs):

        # get sim
        if sim is None:
            sim = MjSim(load_model_from_path(model_path))

        # establish keys
        self.obs_keys = obs_keys.copy()
        self.rwd_keys = rwd_keys
        if sim.model.na>0 and 'act' not in self.obs_keys:
            self.obs_keys.append('act')

        # ids
        self.tip_sids = []
        self.target_sids = []
        for site in sites:
            self.tip_sids.append(sim.model.site_name2id(site))
            self.target_sids.append(sim.model.site_name2id(site+'_target'))

        # configure action space
        self.act_mid = np.mean(sim.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(sim.model.actuator_ctrlrange[:,1]-sim.model.actuator_ctrlrange[:,0])
        self.normalize_act = normalize_act

        # get env
        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)
        self.obs_dict = {}
        self.rwd_dict = {}
        mujoco_env.MujocoEnv.__init__(self, sim=sim, frame_skip=5)
        if self.normalize_act:
            self.action_space.high = np.ones_like(sim.model.actuator_ctrlrange[:,1])
            self.action_space.low  = -1.0 * np.ones_like(sim.model.actuator_ctrlrange[:,0])

    # step the simulation forward
    def step(self, a):
        # apply action and step
        a = np.clip(a, a_min=self.action_space.low, a_max=self.action_space.high)
        if self.normalize_act:
            a = self.act_mid + a*self.act_rng
        self.do_simulation(a, self.frame_skip)

        # observation and rewards
        obs = self.get_obs()
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()
        return obs, env_info['rwd_'+RWD_MODE], bool(env_info['done']), env_info


    # use latest obs, rwds to get all info (be careful, information belongs to different timestamps)
    # Its getting called twice. Once in step and sampler calls it as well
    def get_env_infos(self):
        env_info = {
            'time': self.obs_dict['t'][()],
            'rwd_dense': self.rwd_dict['dense'][()],
            'rwd_sparse': self.rwd_dict['sparse'][()],
            'solved': self.rwd_dict['solved'][()],
            'done': self.rwd_dict['done'][()],
            'obs_dict': self.obs_dict.copy(),
            'rwd_dict': self.rwd_dict,
        }
        return env_info

    def reset_model(self, qp=None, qv=None):
        qp = self.init_qpos.copy() if qp is None else qp
        qv = self.init_qvel.copy() if qv is None else qv
        self.set_state(qp, qv)
        self.sim.forward()
        return self.get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5
        self.viewer.vopt.flags[3] = 1 # render actuators


    # evaluate paths and log metrics to logger
    def evaluate_success(self, paths, logger=None):
        num_success = 0
        num_paths = len(paths)
        horizon = self.spec.max_episode_steps # paths could have early termination

        # success if any last 5 steps is solved
        for path in paths:
            if np.sum(path['env_infos']['solved'][-5:], dtype=np.int) > 0:
                num_success += 1
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
            rwd_dense = np.mean([np.sum(p['env_infos']['rwd_dense'])/horizon for p in paths]) # return rwd/step
            logger.log_kv('rwd_sparse', rwd_sparse)
            logger.log_kv('rwd_dense', rwd_dense)
            logger.log_kv('success_rate', success_percentage)

        return success_percentage