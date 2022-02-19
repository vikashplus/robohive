import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestHMS(TestEnvs):
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

if __name__ == '__main__':
    unittest.main()