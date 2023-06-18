""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
from robohive.tests.test_envs import TestEnvs
from robohive import robohive_myo_suite

class TestMyo(TestEnvs):
    def test_envs(self):
        self.check_envs('Myo Suite', robohive_myo_suite)

if __name__ == '__main__':
    unittest.main()