import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestArms(TestEnvs):
    def test_arms(self):
        env_names = [
            'FrankaReachFixed-v0',
            'FrankaReachRandom-v0',
            'FrankaPushFixed-v0',
            'FrankaPushRandom-v0',
            'FetchReachFixed-v0',
            'FetchReachRandom-v0',
            'FrankaPickPlaceFixed-v0',
            'FrankaPickPlaceRandom-v0'
            ]
        self.check_envs('Arms', env_names)

    def test_arms_visual(self):
        env_names = [
            'FrankaReachRandom_vflat-v0',
            'FrankaReachRandom_vr3m18-v0',
            'FrankaReachRandom_vr3m34-v0',
            'FrankaReachRandom_vr3m50-v0',
            ]
        self.check_envs('Arms(visual)', env_names)

if __name__ == '__main__':
    unittest.main()