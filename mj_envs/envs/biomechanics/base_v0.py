from mj_envs.envs import env_base
from mujoco_py import MjViewer
import numpy as np
import gym

class BaseV0(env_base.MujocoEnv):

    def __init__(self, model_path):
        super().__init__(model_path)

    def _setup(self,
            obs_keys:list,
            weighted_reward_keys:dict,
            sites:list = None,
            frame_skip = 10,
            seed = None,
            is_hardware = False,
            config_path = None,
            rwd_viz = False,
            normalize_act = True,
        ):

        if self.sim.model.na>0 and 'act' not in obs_keys:
            obs_keys.append('act')

        # ids
        self.tip_sids = []
        self.target_sids = []
        if sites:
            for site in sites:
                self.tip_sids.append(self.sim.model.site_name2id(site))
                self.target_sids.append(self.sim.model.site_name2id(site+'_target'))
        
        super()._setup(obs_keys=obs_keys, 
                    weighted_reward_keys=weighted_reward_keys, 
                    frame_skip=frame_skip, 
                    seed=seed, 
                    is_hardware=is_hardware,
                    config_path=config_path,
                    rwd_viz=rwd_viz,
                    normalize_act=normalize_act)


    # step the simulation forward
    def step(self, a):
        if self.normalize_act:
            a = 1.0/(1.0+np.exp(-5.0*(a-0.5)))
        return super().step(a=a)

    # def mj_viewer_setup(self):
    #     self.viewer = MjViewer(self.sim)
    #     self.viewer.cam.azimuth = 90
    #     self.sim.forward()
    #     self.viewer.cam.distance = 1.5
    #     self.viewer.vopt.flags[3] = 1 # render actuators ***
