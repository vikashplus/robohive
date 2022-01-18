from time import process_time
import gym
import numpy as np
from mj_envs.envs import env_base
import collections

class rbpMicrowaveBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
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
               obj_init_pose,
               obj_target_pose,
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):

        self.obj_init_pose = obj_init_pose
        self.obj_target_pose = obj_target_pose
        self.obj_pose_size = len(obj_init_pose)

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=40,
                       **kwargs)

        # Correct the initial obj pose.
        self.init_qpos[-self.obj_pose_size:] = self.obj_init_pose


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'][-self.obj_pose_size:] - self.obj_target_pose

        # update the visual keys
        # visual_obs_dict = self.get_visual_obs_dict(sim=sim)
        # obs_dict.update(visual_obs_dict)
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 10

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<1) + (reach_dist<2)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.5),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


class rbpMicrowaveFixed(rbpMicrowaveBase):

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs


class rbpMicrowaveRandom(rbpMicrowaveBase):

    def reset(self):
        self.target_pose = self.np_random.uniform(low=self.sim.model.actuator_ctrlrange[:,0], high=self.sim.model.actuator_ctrlrange[:,1])
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
