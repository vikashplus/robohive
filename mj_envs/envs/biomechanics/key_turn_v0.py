import collections
import numpy as np
import gym

from mj_envs.envs.biomechanics.base_v0 import BaseV0


class KeyTurnFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'key_qpos', 'key_qvel', 'IFtip_approach', 'THtip_approach']
    DEFAULT_RWD_KEYS_AND_WEIGHTS= {
        'key_turn':1.0,
        'IFtip_approach':-100.0,
        'THtip_approach':-100.0,
        'bonus':4.0,
        'penalty':-25.0
    }

    def __init__(self, model_path:str, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__ 
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we 
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path)

        self._setup(**kwargs)


    def _setup(self, 
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, 
               **kwargs):

        self.keyhead_sid = self.sim.model.site_name2id("keyhead")
        self.IF_sid = self.sim.model.site_name2id("IFtip")
        self.TH_sid = self.sim.model.site_name2id("THtip")   
        # self.init_qpos = self.sim.model.key_qpos[0]    

        super()._setup(obs_keys=obs_keys, 
                       weighted_reward_keys=weighted_reward_keys, 
                       **kwargs)

    def get_obs_vec(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:-1].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:-1].copy()
        self.obs_dict['key_qpos'] = np.array([self.sim.data.qpos[-1]])
        self.obs_dict['key_qvel'] = np.array([self.sim.data.qvel[-1]])
        self.obs_dict['IFtip_approach'] = self.sim.data.site_xpos[self.keyhead_sid]-self.sim.data.site_xpos[self.IF_sid]
        self.obs_dict['THtip_approach'] = self.sim.data.site_xpos[self.keyhead_sid]-self.sim.data.site_xpos[self.TH_sid]

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-1].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-1].copy()
        obs_dict['key_qpos'] = np.array([sim.data.qpos[-1]])
        obs_dict['key_qvel'] = np.array([sim.data.qvel[-1]])
        obs_dict['IFtip_approach'] = sim.data.site_xpos[self.keyhead_sid]-sim.data.site_xpos[self.IF_sid]
        obs_dict['THtip_approach'] = sim.data.site_xpos[self.keyhead_sid]-sim.data.site_xpos[self.TH_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        IF_approach_dist = np.abs(np.linalg.norm(self.obs_dict['IFtip_approach'], axis=-1)-0.040)
        TH_approach_dist = np.abs(np.linalg.norm(self.obs_dict['THtip_approach'], axis=-1)-0.040)

        key_qvel = obs_dict['key_qvel'][:,:,0] if obs_dict['key_qvel'].ndim==3 else obs_dict['key_qvel'][0]
        far_th = .07

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('key_turn', key_qvel),
            ('IFtip_approach', IF_approach_dist),
            ('THtip_approach', TH_approach_dist),
            ('bonus', (key_qvel>np.pi*2) + (key_qvel>np.pi*3)),
            ('penalty', (IF_approach_dist>far_th/2)+(TH_approach_dist>far_th/2) ),
            # Must keys
            ('sparse', key_qvel),
            ('solved', obs_dict['key_qpos']>2*np.pi),
            ('done', (IF_approach_dist>far_th) or (TH_approach_dist>far_th)),
        ))

        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


class KeyTurnRandomEnvV0(KeyTurnFixedEnvV0):

    def reset(self):
        # randomize init pos
        jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
        obs = super().reset(reset_qpos=jnt_init)
        return obs