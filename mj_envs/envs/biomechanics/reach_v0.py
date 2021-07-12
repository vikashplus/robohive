from mj_envs.envs.biomechanics.base_v0 import BaseV0
import numpy as np
import collections

class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self,
                obs_keys:list = DEFAULT_OBS_KEYS,
                rwd_keys_wt:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
                target_reach_range:dict = {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
                **kwargs):
        self.target_reach_range = target_reach_range
        super().__init__(obs_keys=obs_keys, rwd_keys_wt=rwd_keys_wt, sites=self.target_reach_range.keys(), **kwargs)

    def get_obs_vec(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        far_th = .35*len(self.tip_sids)

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<.010) + (reach_dist<.005)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.005),
            ('done',    reach_dist > far_th),
        ))

        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        for site, span in self.target_reach_range.items():
            sid =  self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.np_random.uniform(high=span[0], low=span[1])
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs