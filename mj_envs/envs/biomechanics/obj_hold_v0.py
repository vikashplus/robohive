from mj_envs.envs.biomechanics.base_v0 import BaseV0
import numpy as np
import collections
from mjrl.envs.mujoco_env import get_sim

class ObjHoldFixedEnvV0(BaseV0):

    def __init__(self,
                obs_keys:list = ['hand_qpos', 'hand_qvel', 'obj_pos', 'obj_err'],
                rwd_keys:list = ['goal_dist', 'bonus', 'penalty'],
                **kwargs):

        self.sim = get_sim(model_path=kwargs['model_path'])
        self.object_sid = self.sim.model.site_name2id("object")
        self.goal_sid = self.sim.model.site_name2id("goal")
        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys, sim=self.sim, rwd_viz=False, **kwargs)

    def get_obs(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.data.qpos[:-7].copy()
        self.obs_dict['hand_qvel'] = self.data.qvel[:-6].copy()
        self.obs_dict['obj_pos'] = self.data.site_xpos[self.object_sid]
        self.obs_dict['obj_err'] = self.data.site_xpos[self.goal_sid] - self.data.site_xpos[self.object_sid]
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(np.linalg.norm(self.obs_dict['obj_err'], axis=-1)-0.040)
        gaol_th = .010
        drop = goal_dist > 0.300

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('goal_dist', -100*goal_dist),
            ('bonus', 4.0*(goal_dist<2*gaol_th) + 4.0*(goal_dist<gaol_th)),
            ('penalty', -10*drop),
            # Must keys
            ('sparse', -goal_dist),
            ('solved', goal_dist<gaol_th),
            ('done', drop),
        ))
        # rwd_dict['done'] = False
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict


class ObjHoldRandomEnvV0(ObjHoldFixedEnvV0):

    def reset(self):
        # randomize target pos
        self.model.site_pos[self.goal_sid] = np.array([-.2, -.2, 1]) + self.np_random.uniform(high=np.array([0.030, 0.030, 0.030]), low=np.array([-.030, -.030, -.030]))
        obs = super().reset_model()
        return obs