import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from mj_envs.utils.quatmath import *
import os
from mj_envs.utils.obj_vec_dict import ObsVecDict
import collections
import gym
import time as timer

ADD_BONUS_REWARDS = True
OBS_KEYS = ['object_pos'] # DAPG
RWD_KEYS = ['none'] # DAPG

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
        self.objects_id = [self.knife, self.spatula, self.ratchet, self.screwDriver, self.turner]
        self.curr_obj = self.knife

        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)
        self.obs_dict = {}
        self.rwd_dict = {}
        mujoco_env.MujocoEnv.__init__(self, sim=sim, frame_skip=5)

    # step the simulation forward
    def step(self, a):
        # apply action and step

        a = np.clip(a, -1.0, 1.0)
        # a = self.act_mid + a*self.act_rng
        self.sim.model.body_pos[self.curr_obj] += np.random.uniform(high=[0.02, 0.02, 0.0], low=[-0.02, -0.02, 0.0])
        self.sim.model.body_quat[self.curr_obj] += np.random.uniform(high=[0.05, 0.05, 0, 0], low=[-0.05, -0.05, 0, 0])

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

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('none',    0.0),
            ('done', False),
        ))
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in RWD_KEYS], axis=0)
        return rwd_dict

    def get_obs(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['object_pos'] = self.sim.model.body_pos[self.curr_obj]
        t, obs = self.obsdict2obsvec(self.obs_dict, OBS_KEYS)
        return obs

    def reset_model(self):
        self.sim.reset()
        self.curr_obj = self.np_random.choice(self.objects_id)
        for obj_id in self.objects_id :
            if obj_id == self.curr_obj:
                self.sim.model.body_pos[obj_id] = self.np_random.uniform(high=[0.2, 0.2, 0.045], low=[-0.2, -0.2, 0.045])
                self.sim.model.body_quat[obj_id] = self.np_random.uniform(high=[1.0, 1.0, 0, 0], low=[-1.0, -1.0, 0, 0])
            else :
                self.sim.model.body_pos[obj_id] = self.np_random.uniform(high=[20, 20, 0.45], low=[20, 20, 0.045])
        self.sim.forward()
        return self.get_obs()

    def get_env_infos(self):
        env_info = {
            'time': self.obs_dict['t'][()],
            'rwd_dense': self.rwd_dict['dense'][()],
            # 'rwd_sparse': self.rwd_dict['sparse'][()],
            # 'solved': self.rwd_dict['solved'][()],
            'done': self.rwd_dict['done'][()],
            'obs_dict': self.obs_dict,
            'rwd_dict': self.rwd_dict,
        }
        return env_info

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

if __name__ == '__main__':
    env = gym.make("mixtools-v0")
    
    # Random policy
    class rand_policy():
        def __init__(self, env):
            self.env = env
        def get_action(self, obs):
            return [self.env.action_space.sample()]

    env.reset()

    policy = rand_policy(env)
    # env.visualize_policy_offscreen(policy, horizon=200, num_episodes=2, mode='exploration', save_loc='./')
    env.visualize_policy(policy, horizon=50, num_episodes=2, mode='exploration')
    print(env)
