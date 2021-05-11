from mj_envs.envs.biomechanics.base_v0 import FingerBaseV0
import numpy as np
import collections
from mjrl.envs.mujoco_env import get_sim

class KeyTurnFixedEnvV0(FingerBaseV0):

    def __init__(self,
                obs_keys:list = ['hand_qpos', 'hand_qvel', 'key_qpos', 'key_qvel', 'IFtip_approach', 'THtip_approach'],
                rwd_keys:list = ['key_turn', 'IFtip_approach', 'THtip_approach', 'bonus', 'penalty'],
                **kwargs):

        self.sim = get_sim(model_path=kwargs['model_path'])
        self.keyhead_sid = self.sim.model.site_name2id("keyhead")
        self.IF_sid = self.sim.model.site_name2id("IFtip")
        self.TH_sid = self.sim.model.site_name2id("THtip")
        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys, sim=self.sim, rwd_viz=False, **kwargs)
        # self.init_qpos = self.sim.model.key_qpos[0]

    def get_obs(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.data.qpos[:-1].copy()
        self.obs_dict['hand_qvel'] = self.data.qvel[:-1].copy()
        self.obs_dict['key_qpos'] = np.array([self.data.qpos[-1]])
        self.obs_dict['key_qvel'] = np.array([self.data.qvel[-1]])
        self.obs_dict['IFtip_approach'] = self.data.site_xpos[self.keyhead_sid]-self.data.site_xpos[self.IF_sid]
        self.obs_dict['THtip_approach'] = self.data.site_xpos[self.keyhead_sid]-self.data.site_xpos[self.TH_sid]

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        IF_approach_dist = np.abs(np.linalg.norm(self.obs_dict['IFtip_approach'], axis=-1)-0.040)
        TH_approach_dist = np.abs(np.linalg.norm(self.obs_dict['THtip_approach'], axis=-1)-0.040)
        key_qvel = obs_dict['key_qvel']
        far_th = .07

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('key_turn', obs_dict['key_qvel']),
            ('IFtip_approach', -100*IF_approach_dist),
            ('THtip_approach', -100*TH_approach_dist),
            ('bonus', 4.0*(key_qvel>np.pi*2) + 4.0*(key_qvel>np.pi*3)),
            ('penalty', -25.*(IF_approach_dist>far_th/2)-25.*(TH_approach_dist>far_th/2) ),
            # Must keys
            ('sparse', obs_dict['key_qvel']),
            ('solved', obs_dict['key_qpos']>2*np.pi),
            ('done', (IF_approach_dist>far_th) or (TH_approach_dist>far_th)),
        ))
        # rwd_dict['done'] = False
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict


class KeyTurnRandomEnvV0(KeyTurnFixedEnvV0):

    def reset(self):
        # randomize init pos
        jnt_init = self.np_random.uniform(high=self.model.jnt_range[:,1], low=self.model.jnt_range[:,0])
        obs = super().reset_model(qp=jnt_init)
        return obs