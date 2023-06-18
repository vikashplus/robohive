import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_fm_suite

class TestFM(TestEnvs):
    def test_envs(self):
        self.check_envs('FM Suite', robohive_fm_suite)

if __name__ == '__main__':
    unittest.main()