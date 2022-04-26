import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestClaws(TestEnvs):
    def test_claws(self):
        env_names = [
            'TrifingerReachFixed-v0',
            'TrifingerReachRandom-v0',
            ]
        self.check_envs('Claws', env_names)

if __name__ == '__main__':
    unittest.main()