import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_multitask_suite

class TestKitchen(TestEnvs):

    def test_envs(self):
        self.check_envs('Multi-Task Suite', robohive_multitask_suite)

if __name__ == '__main__':
    unittest.main()