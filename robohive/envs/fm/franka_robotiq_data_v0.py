""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
import numpy as np
from robohive.envs import env_base
from robohive.physics.sim_scene import SimScene
from robohive.utils.xml_utils import reassign_parent
import os
import collections

class FrankaRobotiqData(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'time'          # dummy key
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "none": -0.0,   # dummy key
    }
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Process model to use DManus as end effector
        raw_sim = SimScene.get_sim(curr_dir+model_path)
        raw_xml = raw_sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
        processed_model_path = curr_dir+model_path[:-4]+"_processed.xml"
        with open(processed_model_path, 'w') as file:
            file.write(processed_xml)

        # Process model to use DManus as end effector
        if obsd_model_path == model_path:
            processed_obsd_model_path = processed_model_path
        elif obsd_model_path:
            raw_sim = SimScene.get_sim(curr_dir+obsd_model_path)
            raw_xml = raw_sim.model.get_xml()
            processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
            processed_obsd_model_path = curr_dir+obsd_model_path[:-4]+"_processed.xml"
            with open(processed_obsd_model_path, 'w') as file:
                file.write(processed_xml)
        else:
            processed_obsd_model_path = None

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
        super().__init__(model_path=processed_model_path, obsd_model_path=processed_obsd_model_path, seed=seed)
        os.remove(processed_model_path)
        if processed_obsd_model_path and processed_obsd_model_path!=processed_model_path:
            os.remove(processed_obsd_model_path)

        self._setup(**kwargs)


    def _setup(self,
                nq_arm,
                nq_ee,
                name_ee,
                obs_keys=DEFAULT_OBS_KEYS,
                weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
                **kwargs):

        self.nq_arm = nq_arm
        self.nq_ee = nq_ee
        self.ee_sid = self.sim.model.site_name2id(name_ee)

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=40,
                       **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos[:self.nq_arm+self.nq_ee].copy()

        obs_dict['qp_arm'] = sim.data.qpos[:self.nq_arm].copy()
        obs_dict['qv_arm'] = sim.data.qvel[:self.nq_arm].copy()

        obs_dict['qp_ee'] = sim.data.qpos[self.nq_arm:self.nq_arm+self.nq_ee].copy()
        obs_dict['qv_ee'] = sim.data.qvel[self.nq_arm:self.nq_arm+self.nq_ee].copy()

        obs_dict['pos_ee'] = sim.data.site_xpos[self.ee_sid].copy()
        obs_dict['rot_ee'] = sim.data.site_xmat[self.ee_sid].copy()

        return obs_dict


    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('none',   0.0),
            # Must keys
            ('sparse',  0.0),
            ('solved',  False),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict