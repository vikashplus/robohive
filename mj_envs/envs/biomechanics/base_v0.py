from mj_envs.envs import env_base
from mujoco_py import MjViewer
import numpy as np

class BaseV0(env_base.MujocoEnv):

    def __init__(self, sim=None, sim_obsd=None, model_path=None, # model details
                    sites:tuple = None, # fingertip sites of interests
                    obs_keys:list=None,
                    frame_skip = 10,
                    **kwargs):

        # get sim
        if sim is None:
            sim = env_base.get_sim(model_path=model_path)
            sim_obsd = env_base.get_sim(model_path=model_path)

        if sim.model.na>0 and 'act' not in obs_keys:
            obs_keys.append('act')

        # ids
        self.tip_sids = []
        self.target_sids = []
        if sites:
            for site in sites:
                self.tip_sids.append(sim.model.site_name2id(site))
                self.target_sids.append(sim.model.site_name2id(site+'_target'))

        # get env
        env_base.MujocoEnv.__init__(self,
                                sim = sim,
                                sim_obsd = sim_obsd,
                                model_path = model_path,
                                frame_skip = frame_skip,
                                config_path = None,
                                obs_keys = obs_keys,
                                rwd_mode = "dense",
                                act_mode = "pos",
                                act_normalized = True,
                                is_hardware = False,
                                **kwargs)


    # step the simulation forward
    def step(self, a):
        if self.act_normalized:
            a = 1.0/(1.0+np.exp(-5.0*(a-0.5)))
        return super().step(a=a)

    # def mj_viewer_setup(self):
    #     self.viewer = MjViewer(self.sim)
    #     self.viewer.cam.azimuth = 90
    #     self.sim.forward()
    #     self.viewer.cam.distance = 1.5
    #     self.viewer.vopt.flags[3] = 1 # render actuators ***
