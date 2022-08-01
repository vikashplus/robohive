""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import gym

from mj_envs.envs.myo.base_v0 import BaseV0
from mj_envs.utils.quat_math import mat2euler, euler2quat

class ReorientEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 100.0,
        "rot_dist": 1.0,
        "bonus": 4.0,
        "act_reg": 1,
        "penalty": 10,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)


    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pos_rand_th = 0,
            rot_rand_th = 0,
            **kwargs,
        ):
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.goal_init_pos = self.sim.data.site_xpos[self.goal_sid].copy()
        self.pos_rand_th = pos_rand_th
        self.rot_rand_th = rot_rand_th

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-7] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # place palm up

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid] + np.array([0.1, 0, 0]) # correct for .1m offset between target and object
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos']
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pos_th = .025
        rot_th = 0.262 # 15deg
        drop_th = .200
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = pos_dist > drop_th

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pos_dist', -1.*pos_dist),
            ('rot_dist', -1.*rot_dist),
            ('bonus', 1.*(pos_dist<2*pos_th) + 1.*(pos_dist<pos_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*drop),
            # Must keys
            ('sparse', -rot_dist),
            ('solved', rot_dist<rot_th),
            ('done', drop),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        # Sucess Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        return rwd_dict

    def reset(self):
        # randomize goal
        if self.pos_rand_th==0 and self.rot_rand_th==0:
            self.sim.model.body_pos[self.goal_bid] = self.goal_init_pos
            self.sim.model.body_quat[self.goal_bid] = euler2quat(np.array([0, 0, 1.57]))
        else:
            self.sim.model.body_pos[self.goal_bid] = self.goal_init_pos + self.pos_rand_th*self.np_random.uniform(high=np.array([1, 1, 1]), low=np.array([-1, -1, -1]))
            self.sim.model.body_quat[self.goal_bid] = euler2quat(self.rot_rand_th*self.np_random.uniform(high=np.array([1, 1, 1]), low=np.array([-1, -1, -1])))

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs