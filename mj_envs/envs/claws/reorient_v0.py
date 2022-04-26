""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from mj_envs.envs import env_base
from mj_envs.utils.quat_math import mat2euler, euler2quat

class ReorientBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'reach_pos_err', 'reach_rot_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_pos": -1.0,
        "reach_rot": -0.01,
        "bonus": 4.0,
        "penalty": -50,
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
               object_site_name,
               target_site_name,
               target_xyz_range,
               target_euler_range,
               frame_skip = 40,
               reward_mode = "dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        # ids
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.target_euler_range = target_euler_range

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
        obs_dict['reach_pos_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.object_sid]
        obs_dict['reach_rot_err'] = mat2euler(np.reshape(sim.data.site_xmat[self.target_sid],(3,3))) - mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_pos_dist = np.linalg.norm(obs_dict['reach_pos_err'], axis=-1)
        reach_rot_dist = np.linalg.norm(obs_dict['reach_rot_err'], axis=-1)
        far_th = 1.0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach_pos',   reach_pos_dist),
            ('reach_rot',   reach_rot_dist),
            ('bonus',   (reach_pos_dist<.1) + (reach_pos_dist<.05) + (reach_rot_dist<.3) + (reach_rot_dist<.1)),
            ('penalty', (reach_pos_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_pos_dist-1.0*reach_rot_dist),
            ('solved',  (reach_pos_dist<.050) and (reach_rot_dist<.1)),
            ('done',    reach_pos_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        desired_pos = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        self.sim.model.site_pos[self.target_sid] = desired_pos
        self.sim_obsd.model.site_pos[self.target_sid] = desired_pos

        desired_orien = np.zeros(3)
        desired_orien = self.np_random.uniform(high=self.target_euler_range['high'], low=self.target_euler_range['low'])
        self.sim.model.site_quat[self.target_sid] = euler2quat(desired_orien)
        self.sim_obsd.model.site_quat[self.target_sid] = euler2quat(desired_orien)

        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
