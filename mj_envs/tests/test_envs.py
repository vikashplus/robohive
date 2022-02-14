""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import unittest

import gym
import mj_envs
import numpy as np

class TestEnvs(unittest.TestCase):

    def check_envs(self, module_name, env_names, lite=False, seed=1234):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
            print("Testing env: ", env_name)
            # test init
            env = gym.make(env_name)
            env.seed(seed)

            # test reset
            env.env.reset()
            # test obs vec
            obs = env.env.get_obs()

            if not lite:
                # test obs dict
                obs_dict = env.env.get_obs_dict(env.env.sim)
                # test rewards
                rwd = env.env.get_reward_dict(obs_dict)

                # test vector => dict upgrade
                # print(env.env.get_obs() - env.env.get_obs_vec())
                # assert (env.env.get_obs() == env.env.get_obs_vec()).all(), "check vectorized computations"

            # test env infos
            infos = env.env.get_env_infos()

            # test step (everything together)
            observation, _reward, done, _info = env.env.step(np.zeros(env.env.sim.model.nu))
            del(env)

    # Myo
    def test_myo(self):
        env_names = [
            'motorFingerReachFixed-v0', 'motorFingerReachRandom-v0',
            'myoFingerReachFixed-v0', 'myoFingerReachRandom-v0',
            'myoHandReachFixed-v0', 'myoHandReachRandom-v0',

            'motorFingerPoseFixed-v0', 'motorFingerPoseRandom-v0',
            'myoFingerPoseFixed-v0', 'myoFingerPoseRandom-v0',

            'myoElbowPose1D1MRandom-v0', 'myoElbowPose1D6MRandom-v0',
            'myoElbowPose1D6MExoRandom-v0', 'myoElbowPose1D6M_SoftExo_Random-v0',
            'myoHandPoseFixed-v0', 'myoHandPoseRandom-v0',

            'myoHandKeyTurnFixed-v0', 'myoHandKeyTurnRandom-v0',
            'myoHandObjHoldFixed-v0', 'myoHandObjHoldRandom-v0',
            'myoHandPenTwirlFixed-v0', 'myoHandPenTwirlRandom-v0',

            'myoHandBaodingFixed-v1', 'myoHandBaodingRandom-v1',
            'myoHandBaodingFixed4th-v1','myoHandBaodingFixed8th-v1',
        ]
        for k in range(10): env_names+=['myoHandPose'+str(k)+'Fixed-v0']

        self.check_envs('Myo', env_names)

    # Arms
    def test_arms(self):
        env_names = [
            'FrankaReachFixed-v0',
            'FrankaReachRandom-v0',
            'FrankaPushFixed-v0',
            'FrankaPushRandom-v0',
            'FetchReachFixed-v0',
            'FetchReachRandom-v0'
            ]
        self.check_envs('Arms', env_names)

    # Hand Manipulation Suite
    def test_hand_manipulation_suite(self):
        env_names = [
        'pen-v0',
        'door-v0',
        'hammer-v0',
        'relocate-v0',
        'baoding-v1', 'baoding4th-v1', 'baoding8th-v1'
        ]
        self.check_envs('Hand Manipulation', env_names, lite=True)

    def test_fm(self):
        env_names = [
        'rpFrankaDmanusPoseFixed-v0',
        'rpFrankaDmanusPoseRandom-v0',
        ]
        self.check_envs('FM', env_names)

    def test_kitchen(self):
        env_names = [
        'kitchen-v3',
        'kitchen_close-v3',
        'kitchen_micro_open-v3',
        'kitchen_micro_close-v3',
        'kitchen_rdoor_open-v3',
        'kitchen_rdoor_close-v3',
        'kitchen_ldoor_open-v3',
        'kitchen_ldoor_close-v3',
        'kitchen_sdoor_open-v3',
        'kitchen_sdoor_close-v3',
        'kitchen_light_on-v3',
        'kitchen_light_off-v3',
        'kitchen_knob4_on-v3',
        'kitchen_knob4_off-v3',
        'kitchen_knob3_on-v3',
        'kitchen_knob3_off-v3',
        'kitchen_knob2_on-v3',
        'kitchen_knob2_off-v3',
        'kitchen_knob1_on-v3',
        'kitchen_knob1_off-v3',
        'franka_micro_open-v3','franka_micro_close-v3','franka_micro_random-v3',
        'franka_slide_open-v3','franka_slide_close-v3','franka_slide_random-v3',
        ]
        self.check_envs('Kitchen', env_names, lite=False)


if __name__ == '__main__':
    unittest.main()