""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
import click
import click.testing
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_myo_suite

class TestMyo(TestEnvs):
    def test_envs(self):
        self.check_envs('Myo Suite', robohive_myo_suite)

    def test_myodex(self):
        env_names="MyoHandAirplaneFly-v0,MyoHandAirplaneLift-v0,MyoHandAirplanePass-v0,MyoHandAlarmclockLift-v0,MyoHandAlarmclockPass-v0,MyoHandAlarmclockSee-v0,MyoHandAppleLift-v0,MyoHandApplePass-v0,MyoHandBananaPass-v0,MyoHandBinocularsPass-v0,MyoHandCupDrink-v0,MyoHandCupPass-v0,MyoHandCupPour-v0,MyoHandDuckInspect-v0,MyoHandDuckLift-v0,MyoHandDuckPass-v0,MyoHandElephantLift-v0,MyoHandElephantPass-v0,MyoHandEyeglassesPass-v0,MyoHandFlashlightLift-v0,MyoHandFlashlight1On-v0,MyoHandFlashlight2On-v0,MyoHandFlashlightPass-v0,MyoHandFlutePass-v0,MyoHandFryingpanCook-v0,MyoHandHammerPass-v0,MyoHandHammerUse-v0,MyoHandHandInspect-v0,MyoHandHandPass-v0,MyoHandHeadphonesPass-v0,MyoHandKnifeChop-v0,MyoHandKnifeLift-v0,MyoHandLightbulbPass-v0,MyoHandMouseLift-v0,MyoHandMouseUse-v0,MyoHandMugDrink3-v0,MyoHandPiggybankUse-v0,MyoHandScissorsUse-v0,MyoHandSpheremediumLift-v0,MyoHandStampStamp-v0,MyoHandStanfordbunnyInspect-v0,MyoHandStaplerLift-v0,MyoHandToothbrushLift-v0,MyoHandToothpasteLift-v0,MyoHandToruslargeInspect-v0,MyoHandTrainPlay-v0,MyoHandWatchLift-v0,MyoHandWaterbottleLift-v0,MyoHandWaterbottleShake-v0,MyoHandWineglassDrink1-v0,MyoHandWineglassDrink2-v0,MyoHandWineglassLift-v0,MyoHandWineglassPass-v0".split(",")

        # Check the envs
        self.check_envs('MyoDex', env_names)

        # Check trajectory playback
        from robohive.envs.tcdm.playback_traj import playback_default_traj
        for env in env_names:
            print(f"Testing reference motion playback on: {env}")
            runner = click.testing.CliRunner()
            result = runner.invoke(playback_default_traj, ["--env_name", env, \
                                                        "--horizon", -1, \
                                                        "--num_playback", 1, \
                                                        "--render", "none"])

    def test_myomimic(self):
        env_names=['MyoLegJump-v0', 'MyoLegLunge-v0', 'MyoLegSquat-v0', 'MyoLegLand-v0', 'MyoLegRun-v0', 'MyoLegWalk-v0']
        # Check the envs
        self.check_envs('MyoDex', env_names)

        # Check trajectory playback
        from robohive.envs.tcdm.playback_traj import playback_default_traj
        for env in env_names:
            print(f"Testing reference motion playback on: {env}")
            runner = click.testing.CliRunner()
            result = runner.invoke(playback_default_traj, ["--env_name", env, \
                                                        "--horizon", -1, \
                                                        "--num_playback", 1, \
                                                        "--render", "none"])


if __name__ == '__main__':
    unittest.main()


