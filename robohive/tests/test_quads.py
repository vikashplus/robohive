import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_quad_suite

class TestQuads(TestEnvs):
    def test_envs(self):
        self.check_envs('Quadruped Suite', robohive_quad_suite)

if __name__ == '__main__':
    unittest.main()