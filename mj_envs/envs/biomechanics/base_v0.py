from mj_envs.envs import env_base
from mujoco_py import MjViewer
import numpy as np
import gym

class BaseV0(env_base.MujocoEnv):
    
    muscle_condition = ''
    
    which_muscles = []
    which_gain_muscles = []
    MVC_rest = []
    f_load = {}

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
        
        # TODO: pick it from kwargs
        self.which_muscles = [2, 3, 4]
        self.which_gain_muscles = [100, 100, 100]
        # self.muscle_condition = muscle_condition
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == 'weakness':
            for mus_idx in self.which_muscles:
                self.sim.model.actuator_gainprm[mus_idx,2] = self.which_gain_muscles[mus_idx]*self.sim.model.actuator_gainprm[mus_idx,2]
        # for muscle fatigue we used the model from   
        # Liang Ma, Damien Chablat, Fouad Bennis, Wei Zhang 
        # A new simple dynamic muscle fatigue model and its validation 
        # International Journal of Industrial Ergonomics 39 (2009) 211–220
        elif self.muscle_condition == 'fatigue':
            self.f_load = {}
            self.MVC_rest = {}
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                self.f_load[mus_idx] = []
                self.MVC_rest[mus_idx] = self.sim.model.actuator_gainprm[mus_idx,2]

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
        
        if self.muscle_condition == 'fatigue':
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx,1].copy())
                # import ipdb; ipdb.set_trace()
                f_int = np.sum(self.f_load[mus_idx])/self.MVC_rest[mus_idx]
                f_cem = self.MVC_rest[mus_idx]*np.exp(f_int)
                self.sim.model.actuator_gainprm[mus_idx,2] = f_cem
                self.sim_obsd.model.actuator_gainprm[mus_idx,2] = f_cem

        return super().step(a=a)

    # def mj_viewer_setup(self):
    #     self.viewer = MjViewer(self.sim)
    #     self.viewer.cam.azimuth = 90
    #     self.sim.forward()
    #     self.viewer.cam.distance = 1.5
    #     self.viewer.vopt.flags[3] = 1 # render actuators ***
