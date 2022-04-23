import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestHMS(TestEnvs):
    # Hand Manipulation Suite
    def test_hand_manipulation_suite(self):
        env_names = [
        'pen-v1',
        'door-v1',
        'hammer-v1',
        'relocate-v1',
        'baoding-v1', 'baoding4th-v1', 'baoding8th-v1'
        ]
        self.check_envs('Hand Manipulation', env_names)

    def test_dexman_suite(self):
        env_names = [
            'AdroitAirplaneTrackFixed-v0',
            'AdroitAlarmclockTrackFixed-v0',
            'AdroitAppleTrackFixed-v0',
            'AdroitBananaTrackFixed-v0',
            'AdroitBinocularsTrackFixed-v0',
            'AdroitBowlTrackFixed-v0',
            'AdroitCameraTrackFixed-v0',
            'AdroitCoffeemugTrackFixed-v0',
            'AdroitCubelargeTrackFixed-v0',
            'AdroitCubemediumTrackFixed-v0',
            'AdroitCubemiddleTrackFixed-v0',
            'AdroitCubesmallTrackFixed-v0',
            'AdroitCupTrackFixed-v0',
            'AdroitCylinderlargeTrackFixed-v0',
            'AdroitCylindermediumTrackFixed-v0',
            'AdroitCylindersmallTrackFixed-v0',
            'AdroitDoorknobTrackFixed-v0',
            'AdroitDuckTrackFixed-v0',
            'AdroitElephantTrackFixed-v0',
            'AdroitEyeglassesTrackFixed-v0',
            'AdroitFlashlightTrackFixed-v0',
            'AdroitFluteTrackFixed-v0',
            'AdroitFryingpanTrackFixed-v0',
            'AdroitGamecontrollerTrackFixed-v0',
            'AdroitHammerTrackFixed-v0',
            'AdroitHandTrackFixed-v0',
            'AdroitHeadphonesTrackFixed-v0',
            'AdroitHumanTrackFixed-v0',
            'AdroitKnifeTrackFixed-v0',
            'AdroitLightbulbTrackFixed-v0',
            'AdroitMouseTrackFixed-v0',
            'AdroitMugTrackFixed-v0',
            'AdroitPhoneTrackFixed-v0',
            'AdroitPiggybankTrackFixed-v0',
            'AdroitPyramidlargeTrackFixed-v0',
            'AdroitPyramidmediumTrackFixed-v0',
            'AdroitPyramidsmallTrackFixed-v0',
            'AdroitRubberduckTrackFixed-v0',
            'AdroitScissorsTrackFixed-v0',
            'AdroitSpherelargeTrackFixed-v0',
            'AdroitSpheremediumTrackFixed-v0',
            'AdroitSpheresmallTrackFixed-v0',
            'AdroitStampTrackFixed-v0',
            'AdroitStanfordbunnyTrackFixed-v0',
            'AdroitStaplerTrackFixed-v0',
            'AdroitTableTrackFixed-v0',
            'AdroitTeapotTrackFixed-v0',
            'AdroitToothbrushTrackFixed-v0',
            'AdroitToothpasteTrackFixed-v0',
            'AdroitToruslargeTrackFixed-v0',
            'AdroitTorusmediumTrackFixed-v0',
            'AdroitTorussmallTrackFixed-v0',
            'AdroitTrainTrackFixed-v0',
            'AdroitWatchTrackFixed-v0',
            'AdroitWaterbottleTrackFixed-v0',
            'AdroitWineglassTrackFixed-v0',
            'AdroitWristwatchTrackFixed-v0',
        ]
        self.check_envs('DexMan', env_names)

if __name__ == '__main__':
    unittest.main()