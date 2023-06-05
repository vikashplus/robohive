import unittest
from robohive.tests.test_envs import TestEnvs

class TestArms(TestEnvs):
    def test_arms(self):
        env_names = [
            'FrankaReachFixed-v0',
            'FrankaReachRandom-v0',
            'FrankaPushFixed-v0',
            'FrankaPushRandom-v0',
            'FrankaPickPlaceFixed-v0',
            'FrankaPickPlaceRandom-v0'
            ]
        self.check_envs('Arms(Franka)', env_names)

        env_names = [
            'FetchReachFixed-v0',
            'FetchReachRandom-v0',
            ]
        self.check_envs('Arms(Fetch)', env_names)

    def test_arms_visual(self):
        env_names = [
            'FrankaReachRandom_v2d-v0',
            'FrankaPushRandom_v2d-v0',
            'FrankaPickPlaceRandom_v2d-v0'
            ]
        self.check_envs('Arms(Franka)(Visual)', env_names)

        env_names = [
            'FetchReachRandom_v2d-v0',
            ]
        self.check_envs('Arms(Fetch)(Visual)', env_names)

if __name__ == '__main__':
    unittest.main()
