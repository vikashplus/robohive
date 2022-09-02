""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com), Sudeep Dasari (sdasari@andrew.cmu.edu )
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
import numpy as np
from mj_envs.envs import env_base
from mj_envs.utils.xml_utils import reassign_parent
import os
import collections

class TrackEnv(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }
    def __init__(self, object_name, model_path, obsd_model_path=None, seed=None, **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Process model_path to import the right object
        with open(curr_dir+model_path, 'r') as file:
            processed_xml = file.read()
            processed_xml = processed_xml.replace('OBJECT_NAME', object_name)
        processed_model_path = curr_dir+model_path[:-4]+"_processed.xml"
        with open(processed_model_path, 'w') as file:
            file.write(processed_xml)

        # Process obsd_model_path to import the right object
        if obsd_model_path == model_path:
            processed_obsd_model_path = processed_model_path
        elif obsd_model_path:
            with open(curr_dir+obsd_model_path, 'r') as file:
                processed_xml = file.read()
                processed_xml = processed_xml.replace('OBJECT_NAME', object_name)
            processed_obsd_model_path = curr_dir+model_path[:-4]+"_processed.xml"
            with open(processed_obsd_model_path, 'w') as file:
                file.write(processed_xml)
        else:
            processed_obsd_model_path = None

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, object_name, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=processed_model_path, obsd_model_path=processed_obsd_model_path, seed=seed)
        os.remove(processed_model_path)
        if processed_obsd_model_path and processed_obsd_model_path!=processed_model_path:
            os.remove(processed_obsd_model_path)

        self._setup(**kwargs)


    def _setup(self,
               target_pose,
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):

        if isinstance(target_pose, np.ndarray):
            self.target_type = 'fixed'
            self.target_pose = target_pose
        elif target_pose == 'random':
            self.target_type = 'random'
            self.target_pose = self.sim.data.qpos.copy() # fake target for setup
        elif target_pose == None:
            self.target_type = 'track'
            self.target_pose = self.sim.data.qpos.copy() # fake target until we get the reference motion setup

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=40,
                       **kwargs)
        if self.sim.model.nkey>0:
            self.init_qpos[:] = self.sim.model.key_qpos[0,:]


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
        return obs_dict


    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 10

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',   pose_dist),
            ('bonus',   (pose_dist<1) + (pose_dist<2)),
            ('penalty', (pose_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<.5),
            ('done',    pose_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_target_pose(self):
        if self.target_type == 'fixed':
            return self.target_pose
        elif self.target_type == 'random':
            return self.np_random.uniform(low=self.sim.model.actuator_ctrlrange[:,0], high=self.sim.model.actuator_ctrlrange[:,1])
        elif self.target_type == 'track':
            return self.target_pose # ToDo: Update the target pose as per the tracking trajectory

    def reset(self):
        self.target_pose = self.get_target_pose()
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
