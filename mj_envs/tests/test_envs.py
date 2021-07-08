import unittest

import gym
import mj_envs


class TestEnvs(unittest.TestCase):

    def load_envs(self, module_name, env_names):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
           print("Testing env: ", env_name)
           env = gym.make(env_name)

    # Biomechanics
    def test_biomechanics(self):
        env_names = [
            'FingerReachMotorFixed-v0', 'FingerReachMotorRandom-v0',
            'FingerReachMuscleFixed-v0', 'FingerReachMuscleRandom-v0',

            'FingerPoseMotorFixed-v0', 'FingerPoseMotorRandom-v0',
            'FingerPoseMuscleFixed-v0', 'FingerPoseMuscleRandom-v0',

            'IFTHPoseMuscleRandom-v0',
            'IFTHKeyTurnFixed-v0', 'IFTHKeyTurnRandom-v0',

            'IFTHPoseMuscleRandom-v0', 'HandPoseAMuscleFixed-v0',
            'IFTHKeyTurnFixed-v0', 'IFTHKeyTurnRandom-v0',
            'HandObjHoldFixed-v0', 'HandObjHoldRandom-v0',
            'HandPenTwirlFixed-v0', 'HandPenTwirlRandom-v0',
        ]
        self.load_envs('Biomechanics', env_names)

    # Hand Manipulation Suite
    def test_hand_manipulation_suite(self):
        env_names = [
        'pen-v0',
        'door-v0',
        'hammer-v0',
        'relocate-v0',
        ]
        self.load_envs('Hand Manipulation', env_names)

if __name__ == '__main__':
    unittest.main()
