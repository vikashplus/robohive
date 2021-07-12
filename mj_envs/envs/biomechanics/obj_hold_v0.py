from mj_envs.envs.biomechanics.base_v0 import BaseV0
from mj_envs.envs.env_base import get_sim
import numpy as np
import collections

class ObjHoldFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'obj_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal_dist": -100.0,
        "bonus": 4.0,
        "penalty": -10,
    }

    def __init__(self,
                obs_keys:list = DEFAULT_OBS_KEYS,
                rwd_keys_wt:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
                **kwargs):

        self.sim = get_sim(model_path=kwargs['model_path'])
        self.sim_obsd = get_sim(model_path=kwargs['model_path'])
        self.object_sid = self.sim.model.site_name2id("object")
        self.goal_sid = self.sim.model.site_name2id("goal")
        super().__init__(obs_keys=obs_keys, rwd_keys_wt=rwd_keys_wt, sim=self.sim, sim_obsd=self.sim_obsd, rwd_viz=False, **kwargs)

    def get_obs_vec(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:-7].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:-6].copy()
        self.obs_dict['obj_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['obj_err'] = self.sim.data.site_xpos[self.goal_sid] - self.sim.data.site_xpos[self.object_sid]
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['obj_err'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(np.linalg.norm(self.obs_dict['obj_err'], axis=-1)-0.040)
        gaol_th = .010
        drop = goal_dist > 0.300

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('goal_dist', goal_dist),
            ('bonus', (goal_dist<2*gaol_th) + (goal_dist<gaol_th)),
            ('penalty', drop),
            # Must keys
            ('sparse', -goal_dist),
            ('solved', goal_dist<gaol_th),
            ('done', drop),
        ))
        # rwd_dict['done'] = False
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


class ObjHoldRandomEnvV0(ObjHoldFixedEnvV0):

    def reset(self):
        # randomize target pos
        self.sim.model.site_pos[self.goal_sid] = np.array([-.2, -.2, 1]) + self.np_random.uniform(high=np.array([0.030, 0.030, 0.030]), low=np.array([-.030, -.030, -.030]))
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs