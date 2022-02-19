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
            'FetchReachRandom-v0'
            ]
        self.check_envs('Arms', env_names)

if __name__ == '__main__':
    unittest.main()