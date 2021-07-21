import collections
import gym
import numpy as np

from mj_envs.envs import env_base

class PushBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'object_err', 'target_err', 'object_pos', 'target_pos'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -1.0,
        "target_dist": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path, **kwargs):

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
               robot_site_name,
               object_site_name,
               target_site_name,
               target_xyz_range,
               frame_skip=4,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['object_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['target_pos'] = sim.data.site_xpos[self.target_sid]
        obs_dict['object_err'] = sim.data.site_xpos[self.object_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['target_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.object_sid]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        object_dist = np.linalg.norm(obs_dict['object_err'], axis=-1)
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        far_th = 1.25
        if object_dist < 0.1: 
            object_dist = 0.0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('object_dist',   object_dist),
            ('target_dist',   target_dist),
            ('bonus',   (object_dist<.1) + (target_dist<.1) + (target_dist<.05)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*target_dist),
            ('solved',  target_dist<.10),
            ('done',    object_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.sim.data.qpos.ravel().copy()
        qvel = self.sim.data.qvel.ravel().copy()
        object_pos = self.sim.model.body_pos[self.object_sid].copy()
        target_pos = self.sim.model.body_pos[self.target_sid].copy()
        grasp_pos = self.sim.model.body_pos[self.grasp_sid].copy()
        return dict(qpos=qpos, qvel=qvel, object_pos=object_pos, target_pos=target_pos, grasp_pos=grasp_pos)

    def set_env_state(self, state_dict) : 
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        target_pos = state_dict['target_pos']
        object_pos = state_dict['object_pos']
        self.sim.model.body_pos[self.object_sid] = object_pos
        self.sim.model.body_pos[self.target_sid] = target_pos
        self.set_state(qp, qv)
        self.sim.forward()
