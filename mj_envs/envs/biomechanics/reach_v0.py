from mj_envs.envs.biomechanics.base_v0 import FingerBaseV0
import numpy as np
import collections

class ReachEnvV0(FingerBaseV0):

    def __init__(self,
                obs_keys:list = ['qpos', 'qvel', 'tip_pos', 'reach_err'],
                rwd_keys:list = ['reach', 'bonus', 'penalty'],
                target_reach_range:dict = {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
                **kwargs):
        self.target_reach_range = target_reach_range
        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys, sites=self.target_reach_range.keys(),  **kwargs)

    def get_obs(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.data.qvel[:].copy()
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.data.act[:].copy()

        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        far_th = .35*len(self.tip_sids)

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
        for site, span in self.target_reach_range.items():
            sid =  self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.np_random.uniform(high=span[0], low=span[1])
        obs = super().reset()
        return obs