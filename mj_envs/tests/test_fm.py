import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestFM(TestEnvs):
    def test_fm(self):
        env_names = [
        'rpFrankaDmanusPoseFixed-v0',
        'rpFrankaDmanusPoseRandom-v0',
        'rpFrankaRobotiqPoseFixed-v0',
        'rpFrankaRobotiqPoseRandom-v0',
        'rpFrankaRobotiqData-v0'
        ]
        self.check_envs('FM', env_names)

if __name__ == '__main__':
    unittest.main()