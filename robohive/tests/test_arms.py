import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_arm_suite

class TestArms(TestEnvs):
    def test_envs(self):
        self.check_envs('Arm Suite', robohive_arm_suite)

if __name__ == '__main__':
    unittest.main()
