import numpy as np
# from mjrl.envs import mujoco_env
from mj_envs.envs import env_base
# from mujoco_py import MjViewer
import os
import collections

class ReachBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'reach_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }


    def __init__(self, model_path, config_path, robot_site_name, target_site_name, **kwargs):
        # get sims
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.sim = env_base.get_sim(model_path=curr_dir+model_path)
        self.sim_obsd = env_base.get_sim(model_path=curr_dir+model_path)

        # ids
        self.grasp = self.sim.model.site_name2id(robot_site_name)
        self.target = self.sim.model.site_name2id(target_site_name)

        # get env
        env_base.MujocoEnv.__init__(self,
                                sim = self.sim,
                                sim_obsd = self.sim_obsd,
                                frame_skip = 40,
                                config_path = config_path,
                                obs_keys = self.DEFAULT_OBS_KEYS,
                                rwd_keys_wt = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                                rwd_mode = "dense",
                                act_mode = "pos",
                                act_normalized = True,
                                is_hardware = False,
                                **kwargs)


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['reach_err'] = sim.data.site_xpos[self.target]-sim.data.site_xpos[self.grasp]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        far_th = 1.0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<.1) + (reach_dist<.05)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.050),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # def mj_viewer_setup(self):
    #     self.viewer = MjViewer(self.sim)
    #     self.viewer.cam.azimuth = 90
    #     self.sim.forward()
    #     self.viewer.cam.distance = 1.5


class ReachEnvFixed(ReachBase):

    def reset(self):
        self.sim.model.site_pos[self.target] = np.array([0.2, 0.3, 1.2])
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs


class ReachEnvRandom(ReachBase):

    def reset(self):
        self.sim.model.site_pos[self.target] = self.np_random.uniform(high=[0.3, .5, 1.2], low=[-.3, .1, .8])
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
