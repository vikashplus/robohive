import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_hand_suite

class TestHMS(TestEnvs):
    # Hand Manipulation Suite
    def test_hand_manipulation_suite(self):
        self.check_envs('Hand Suite', robohive_hand_suite)

if __name__ == '__main__':
    unittest.main()
