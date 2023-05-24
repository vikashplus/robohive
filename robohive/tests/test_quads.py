import unittest
from robohive.tests.test_envs import TestEnvs

class TestQuadss(TestEnvs):
    def test_claws(self):
        env_names = [
            'DKittyWalkFixed-v0',
            'DKittyWalkRandom-v0',
            'DKittyWalkRandom_v2d-v0',
            'DKittyStandFixed-v0',
            'DKittyStandRandom-v0',
            'DKittyStandRandom_v2d-v0',
            'DKittyOrientFixed-v0',
            'DKittyOrientRandom-v0',
            'DKittyOrientRandom_v2d-v0',
            ]
        self.check_envs('Claws', env_names)

if __name__ == '__main__':
    unittest.main()