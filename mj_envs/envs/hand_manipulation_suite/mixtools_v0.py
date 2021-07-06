import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from mj_envs.utils.quatmath import *
import os
from mj_envs.utils.obj_vec_dict import ObsVecDict
import collections
import gym

ADD_BONUS_REWARDS = True
OBS_KEYS = ['hand_jnt', 'obj_vel', 'palm_pos', 'obj_pos', 'obj_rot', 'target_pos', 'nail_impact'] # DAPG
RWD_KEYS = ['palm_obj', 'tool_target', 'target_goal', 'smooth', 'bonus'] # DAPG

#OBS_KEYS = ['hand_jnt', 'obj_vel', 'palm_pos', 'obj_pos', 'obj_rot', 'target_pos', 'nail_impact', 'tool_pos', 'goal_pos', 'hand_vel']

RWD_MODE = 'dense' # dense/ sparse

class MixToolsEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, ObsVecDict):
    def __init__(self, *args, **kwargs):
        # get sim
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        sim = mujoco_env.get_sim(model_path=curr_dir+'/assets/Adroit_mixtools.xml')
        self.knife = sim.model.body_name2id('knife')
        self.spatula = sim.model.body_name2id('spatula')
        self.ratchet = sim.model.body_name2id('ratchet')
        self.screwDriver = sim.model.body_name2id('screwDriver')
        self.turner = sim.model.body_name2id('turner')

        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)
        self.obs_dict = {}
        self.rwd_dict = {}
        mujoco_env.MujocoEnv.__init__(self, sim=sim, frame_skip=5)

    # step the simulation forward
    def step(self, a):
        # apply action and step
        raise NotImplementedError
        
        a = np.clip(a, -1.0, 1.0)
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


if __name__ == '__main__':
	env = gym.make("mixtools-v0")
	print(env)
