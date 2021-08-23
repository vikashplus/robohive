from mj_envs.envs import env_base
from mujoco_py import MjViewer
import numpy as np
import gym

class BaseV0(env_base.MujocoEnv):

    MVC_rest = []
    f_load = {}

    def __init__(self, model_path):
        super().__init__(model_path)

    def _setup(self,
            obs_keys:list,
            weighted_reward_keys:dict,
            sites:list = None,
            frame_skip = 10,
            seed = None,  # <- doesn't seem to be needed
            is_hardware = False,  # <- doesn't seem to be needed
            config_path = None,  # <- doesn't seem to be needed
            rwd_viz = False,  # <- doesn't seem to be needed
            normalize_act = True, # <- doesn't seem to be needed
            **kwargs,
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

        self.muscle_condition =  kwargs.pop('condition_muscles', '')

        # self.muscle_condition = muscle_condition
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == 'weakness':
            for mus_idx in self.which_muscles:
                self.sim.model.actuator_gainprm[mus_idx,2] = 0.5*self.sim.model.actuator_gainprm[mus_idx,2]

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

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == 'reafferentation':
            self.EPLpos = self.sim.model.actuator_name2id('EPL')
            self.EIPpos = self.sim.model.actuator_name2id('EIP')


        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    frame_skip=frame_skip,
                    seed=seed,# <- doesn't seem to be needed
                    is_hardware=is_hardware, # <- doesn't seem to be needed
                    config_path=config_path, # <- doesn't seem to be needed
                    rwd_viz=rwd_viz, # <- doesn't seem to be needed
                    normalize_act=normalize_act, # <- doesn't seem to be needed
                    **kwargs)


    # step the simulation forward
    def step(self, a):
        if self.normalize_act:
            a = 1.0/(1.0+np.exp(-5.0*(a-0.5)))

        if self.muscle_condition == 'fatigue':
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):

                if self.sim.data.actuator_moment.shape[1]==1:
                    self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx].copy())
                else:
                    self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx,1].copy())

                f_int = np.sum(self.f_load[mus_idx])/self.MVC_rest[mus_idx]
                f_cem = self.MVC_rest[mus_idx]*np.exp(f_int)
                self.sim.model.actuator_gainprm[mus_idx,2] = f_cem
                self.sim_obsd.model.actuator_gainprm[mus_idx,2] = f_cem
        elif self.muscle_condition == 'reafferentation':
            # redirect EIP --> EPL
            a[self.EPLpos] = a[self.EIPpos].copy()
            # Set EIP to 0
            a[self.EIPpos] = 0

        return super().step(a=a)

    # def mj_viewer_setup(self):
    #     self.viewer = MjViewer(self.sim)
    #     self.viewer.cam.azimuth = 90
    #     self.sim.forward()
    #     self.viewer.cam.distance = 1.5
    #     self.viewer.vopt.flags[3] = 1 # render actuators ***
