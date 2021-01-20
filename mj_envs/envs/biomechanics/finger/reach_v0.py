from mj_envs.envs.biomechanics.finger.base_v0 import FingerBaseV0
import numpy as np
import collections

class ReachEnvV0(FingerBaseV0):

    def __init__(self,
                obs_keys:list = ['qpos', 'qvel', 'tip_pos', 'reach_err'],
                rwd_keys:list = ['reach', 'bonus', 'penalty'],
                reach_target_xyz_range = np.array(([0.2, 0.05, 0.20], [0.2, 0.05, 0.20])),
                **kwargs):
        self.reach_target_xyz_range = reach_target_xyz_range
        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys)

    def get_obs(self):
        # qpos for hand, xpos for obj, xpos for target
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.data.qvel[:].copy()
        self.obs_dict['tip_pos'] = self.data.site_xpos[self.tip_sid].copy()
        self.obs_dict['target_pos'] = self.data.site_xpos[self.target_sid].copy()
        self.obs_dict['reach_err'] = self.obs_dict['target_pos']-self.obs_dict['tip_pos']
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        far_th = .35

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*reach_dist),
            ('bonus',   4.0*(reach_dist<.010) + 4.0*(reach_dist<.005)),
            ('penalty', -50.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.005),
            ('done',    reach_dist > far_th),
        ))

        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict

    def reset(self):
        self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.reach_target_xyz_range[0], low=self.reach_target_xyz_range[1])
        obs = super().reset()
        return obs