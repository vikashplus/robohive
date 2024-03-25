""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections

import numpy as np

from robohive.envs import env_base
from robohive.utils import gym
from robohive.utils.quat_math import euler2quat


class PickPlaceV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp_robot', 'qp_object', 'qv_robot', 'qv_object', 'object_err', 'target_err'
    ]
    DEFAULT_PROPRIO_KEYS = [
        'qp_robot', 'qp_object', 'qv_robot', 'qv_object'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -1.0,
        "target_dist": -1.0,
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
               robot_ndof,
               robot_site_name,
               object_site_name,
               target_site_name,
               target_xyz_range,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               randomize=False,
               geom_sizes={'high':[.05, .05, .05], 'low':[.2, 0.2, 0.2]},
               **kwargs,
        ):

        self.robot_ndof = robot_ndof
        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.randomize = randomize
        self.geom_sizes = geom_sizes

        # Save body init pos
        self.init_body_pos = {}
        for body in ["obj0", "obj1", "obj2"]:
            bid = self.sim.model.body_name2id(body)
            self.init_body_pos[body] = self.sim.model.body_pos[bid].copy()

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        self.viewer_setup(distance=1.25, azimuth=-90, elevation=-20)


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_robot'] = sim.data.qpos[:self.robot_ndof].copy()
        obs_dict['qv_robot'] = sim.data.qvel[:self.robot_ndof].copy()
        obs_dict['qp_object'] = sim.data.qpos[self.robot_ndof:].copy()
        obs_dict['qv_object'] = sim.data.qvel[self.robot_ndof:].copy()
        obs_dict['object_err'] = sim.data.site_xpos[self.object_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['target_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.object_sid]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        object_dist = np.linalg.norm(obs_dict['object_err'], axis=-1)
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        far_th = 1.25

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('object_dist',   object_dist),
            ('target_dist',   target_dist),
            ('bonus',   (object_dist<.1) + (target_dist<.1) + (target_dist<.05)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*target_dist),
            ('solved',  target_dist<.050),
            ('done',    object_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, **kwargs):

        if self.randomize:
            # target location
            self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
            self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]

            # object shapes and locations
            for body in ["obj0", "obj1", "obj2"]:
                bid = self.sim.model.body_name2id(body)
                self.sim.model.body_pos[bid] = self.init_body_pos[body] + self.np_random.uniform(low=[-.010, -.010, -.010], high=[-.010, -.010, -.010])# random pos
                self.sim.model.body_quat[bid] = euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)) ) # random quat

                for gid in range(self.sim.model.body_geomnum[bid]):
                    gid+=self.sim.model.body_geomadr[bid]
                    self.sim.model.geom_type[gid]=self.np_random.choice([2,3,4,5,6]) # random shape
                    self.sim.model.geom_size[gid]=self.np_random.uniform(low=self.geom_sizes['low'], high=self.geom_sizes['high']) # random size
                    self.sim.model.geom_pos[gid]=self.np_random.uniform(low=-1*self.sim.model.geom_size[gid], high=self.sim.model.geom_size[gid]) # random pos
                    self.sim.model.geom_quat[gid]=euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)) ) # random quat
                    self.sim.model.geom_rgba[gid]=self.np_random.uniform(low=[.2, .2, .2, 1], high=[.9, .9, .9, 1]) # random color
            self.sim.forward()

        obs = super().reset(self.init_qpos, self.init_qvel, **kwargs)
        return obs

    # def viewer_setup(self):
    #     self.sim.renderer.set_free_camera_settings(
    #             distance=1.25,
    #             azimuth=-90,
    #             elevation=-20,
    #     )