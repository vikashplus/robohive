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
                self.sim.model.actuator_gainprm[mus_idx,2] = self.which_gain_muscles[mus_idx]*sim.model.actuator_gainprm[mus_idx,2]
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
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        """

        # handle fatigue
        if self.muscle_condition == 'fatigue':
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx,1].copy())
                f_int = np.sum(self.f_load[mus_idx])/self.MVC_rest[mus_idx]
                f_cem = self.MVC_rest[mus_idx]*np.exp(f_int)
                self.sim.model.actuator_gainprm[mus_idx,2] = f_cem
                self.sim_obsd.model.actuator_gainprm[mus_idx,2] = f_cem


        # step the env one step forward
        if self.sim.model.na:
            # explicitely project normalized space (-1,1) to actuator space (0,1)
            # TODO: actuator space may not always be (0,1)
            a = 1.0/(1.0+np.exp(-5.0*(a-0.5)))
            isNormalized = False # refuse internal reprojection as we explicitely did it here
        else:
            isNormalized = True # accept internal reprojection as we explicitely did it her

        self.last_ctrl = self.robot.step(ctrl_desired=a,
                                        ctrl_normalized=isNormalized,
                                        step_duration=self.dt,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # returns obs(t+1), rew(t), done(t), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info

    # def mj_viewer_setup(self):
    #     self.viewer = MjViewer(self.sim)
    #     self.viewer.cam.azimuth = 90
    #     self.sim.forward()
    #     self.viewer.cam.distance = 1.5
    #     self.viewer.vopt.flags[3] = 1 # render actuators ***
