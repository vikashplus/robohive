import unittest

import gym
import mj_envs


class TestEnvs(unittest.TestCase):

    def load_envs(self, module_name, env_names):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
           print("Testing env: ", env_name)
           env = gym.make(env_name)

    # Franka Kitchen
    def test_frankakitchen(self):
        env_names = [
            'kitchen-v0',
            'kitchen-v2',
            'kitchen_micro_open-v2',
            'kitchen_rdoor_open-v2',
            'kitchen_ldoor_open-v2',
            'kitchen_sdoor_open-v2',
            'kitchen_light_on-v2',
            'kitchen_knob4_on-v2',
            'kitchen_knob3_on-v2',
            'kitchen_knob2_on-v2',
            'kitchen_knob1_on-v2']
        self.load_envs('Franka Kitchen', env_names)

    # Arms
    def test_arms(self):
        env_names = [
            'FrankaReachFixed-v0',
            'FrankaReachRandom-v0',
            'FetchReachFixed-v0',
            'FetchReachFixed-v0']
        self.load_envs('Arms', env_names)


if __name__ == '__main__':
    unittest.main()
