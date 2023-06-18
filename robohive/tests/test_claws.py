import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_claw_suite

class TestClaws(TestEnvs):
    def test_envs(self):
        self.check_envs('Claw Suite', robohive_claw_suite)

if __name__ == '__main__':
    unittest.main()