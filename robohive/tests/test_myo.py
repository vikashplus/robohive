""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
import click
import click.testing
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_myo_suite
from robohive import robohive_myodex_suite

class TestMyo(TestEnvs):
    def test_myosuite_envs(self):
        self.check_envs('Myo Suite', robohive_myo_suite)

    def test_myodex_envs(self):
        self.check_envs('MyoDex Suite', robohive_myodex_suite)

        # Check trajectory playback
        from robohive.envs.tcdm.playback_traj import playback_default_traj
        for env in robohive_myodex_suite:
            print(f"Testing reference motion playback on: {env}")
            runner = click.testing.CliRunner()
            result = runner.invoke(playback_default_traj, ["--env_name", env, \
                                                        "--horizon", -1, \
                                                        "--num_playback", 1, \
                                                        "--render", "none"])


    def no_test_myomimic(self):
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


