""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
from robohive.tests.test_envs import TestEnvs
from robohive.logger.examine_reference import examine_reference
import click.testing

class TestTCDM(TestEnvs):
    def test_tcdm_ref_suite(self):
        env_names = [
            'AdroitAirplaneFly-v0', 'AdroitAirplanePass-v0', 'AdroitAlarmclockLift-v0', 'AdroitAlarmclockSee-v0', 'AdroitBananaPass-v0', 'AdroitBinocularsPass-v0', 'AdroitCupDrink-v0', 'AdroitCupPour-v0', 'AdroitDuckInspect-v0', 'AdroitDuckLift-v0', 'AdroitElephantPass-v0', 'AdroitEyeglassesPass-v0', 'AdroitFlashlightLift-v0', 'AdroitFlashlightOn-v0', 'AdroitFlutePass-v0', 'AdroitFryingpanCook-v0', 'AdroitHammerUse-v0', 'AdroitHandInspect-v0', 'AdroitHeadphonesPass-v0', 'AdroitKnifeChop-v0', 'AdroitLightbulbPass-v0', 'AdroitMouseLift-v0', 'AdroitMouseUse-v0', 'AdroitMugDrink3-v0', 'AdroitPiggybankUse-v0', 'AdroitScissorsUse-v0', 'AdroitSpheremediumLift-v0', 'AdroitStampStamp-v0', 'AdroitStanfordbunnyInspect-v0', 'AdroitStaplerLift-v0', 'AdroitToothbrushLift-v0', 'AdroitToothpasteLift-v0', 'AdroitToruslargeInspect-v0', 'AdroitTrainPlay-v0', 'AdroitTrainPlay1-v0', 'AdroitWatchLift-v0', 'AdroitWaterbottleLift-v0', 'AdroitWaterbottleShake-v0', 'AdroitWineglassDrink1-v0', 'AdroitWineglassDrink2-v0'        ]
        self.check_envs('TCDM-reference', env_names)

    def test_tcdm_fixed_random_suite(self):
        env_names = [
            'AdroitAirplaneFixed-v0', 'AdroitAirplaneRandom-v0',
            'AdroitAlarmclockFixed-v0', 'AdroitAlarmclockRandom-v0',
            'AdroitAppleFixed-v0', 'AdroitAppleRandom-v0',
            'AdroitBananaFixed-v0', 'AdroitBananaRandom-v0',
            'AdroitBinocularsFixed-v0', 'AdroitBinocularsRandom-v0',
            'AdroitBowlFixed-v0', 'AdroitBowlRandom-v0',
            'AdroitCameraFixed-v0', 'AdroitCameraRandom-v0',
            'AdroitCoffeemugFixed-v0', 'AdroitCoffeemugRandom-v0',
            'AdroitCubelargeFixed-v0', 'AdroitCubelargeRandom-v0',
            'AdroitCubemediumFixed-v0', 'AdroitCubemediumRandom-v0',
            'AdroitCubemiddleFixed-v0', 'AdroitCubemiddleRandom-v0',
            'AdroitCubesmallFixed-v0', 'AdroitCubesmallRandom-v0',
            'AdroitCupFixed-v0', 'AdroitCupRandom-v0',
            'AdroitCylinderlargeFixed-v0', 'AdroitCylinderlargeRandom-v0',
            'AdroitCylindermediumFixed-v0', 'AdroitCylindermediumRandom-v0',
            'AdroitCylindersmallFixed-v0', 'AdroitCylindersmallRandom-v0',
            'AdroitDoorknobFixed-v0', 'AdroitDoorknobRandom-v0',
            'AdroitDuckFixed-v0', 'AdroitDuckRandom-v0',
            'AdroitElephantFixed-v0', 'AdroitElephantRandom-v0',
            'AdroitEyeglassesFixed-v0', 'AdroitEyeglassesRandom-v0',
            'AdroitFlashlightFixed-v0', 'AdroitFlashlightRandom-v0',
            'AdroitFluteFixed-v0', 'AdroitFluteRandom-v0',
            'AdroitFryingpanFixed-v0', 'AdroitFryingpanRandom-v0',
            'AdroitGamecontrollerFixed-v0', 'AdroitGamecontrollerRandom-v0',
            'AdroitHammerFixed-v0', 'AdroitHammerRandom-v0',
            'AdroitHandFixed-v0', 'AdroitHandRandom-v0',
            'AdroitHeadphonesFixed-v0', 'AdroitHeadphonesRandom-v0',
            'AdroitHumanFixed-v0', 'AdroitHumanRandom-v0',
            'AdroitKnifeFixed-v0', 'AdroitKnifeRandom-v0',
            'AdroitLightbulbFixed-v0', 'AdroitLightbulbRandom-v0',
            'AdroitMouseFixed-v0', 'AdroitMouseRandom-v0',
            'AdroitMugFixed-v0', 'AdroitMugRandom-v0',
            'AdroitPhoneFixed-v0', 'AdroitPhoneRandom-v0',
            'AdroitPiggybankFixed-v0', 'AdroitPiggybankRandom-v0',
            'AdroitPyramidlargeFixed-v0', 'AdroitPyramidlargeRandom-v0',
            'AdroitPyramidmediumFixed-v0', 'AdroitPyramidmediumRandom-v0',
            'AdroitPyramidsmallFixed-v0', 'AdroitPyramidsmallRandom-v0',
            'AdroitRubberduckFixed-v0', 'AdroitRubberduckRandom-v0',
            'AdroitScissorsFixed-v0', 'AdroitScissorsRandom-v0',
            'AdroitSpherelargeFixed-v0', 'AdroitSpherelargeRandom-v0',
            'AdroitSpheremediumFixed-v0', 'AdroitSpheremediumRandom-v0',
            'AdroitSpheresmallFixed-v0', 'AdroitSpheresmallRandom-v0',
            'AdroitStampFixed-v0', 'AdroitStampRandom-v0',
            'AdroitStanfordbunnyFixed-v0', 'AdroitStanfordbunnyRandom-v0',
            'AdroitStaplerFixed-v0', 'AdroitStaplerRandom-v0',
            'AdroitTableFixed-v0', 'AdroitTableRandom-v0',
            'AdroitTeapotFixed-v0', 'AdroitTeapotRandom-v0',
            'AdroitToothbrushFixed-v0', 'AdroitToothbrushRandom-v0',
            'AdroitToothpasteFixed-v0', 'AdroitToothpasteRandom-v0',
            'AdroitToruslargeFixed-v0', 'AdroitToruslargeRandom-v0',
            'AdroitTorusmediumFixed-v0', 'AdroitTorusmediumRandom-v0',
            'AdroitTorussmallFixed-v0', 'AdroitTorussmallRandom-v0',
            'AdroitTrainFixed-v0', 'AdroitTrainRandom-v0',
            'AdroitWatchFixed-v0', 'AdroitWatchRandom-v0',
            'AdroitWaterbottleFixed-v0', 'AdroitWaterbottleRandom-v0',
            'AdroitWineglassFixed-v0', 'AdroitWineglassRandom-v0',
            'AdroitWristwatchFixed-v0', 'AdroitWristwatchRandom-v0',
        ]
        self.check_envs('TCDM-Fixed+Random', env_names)

    def test_playback(self):
        # Call your function and test its output/assertions
        print("Testing trajectory playback")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_reference, ["--env_name", "AdroitAirplanePass-v0", \
                                            "--render", "none",])
        print(result.output.strip())
        self.assertEqual(result.exception, None, result.exception)

if __name__ == '__main__':
    unittest.main()