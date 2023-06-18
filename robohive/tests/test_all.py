import unittest
from robohive.tests.test_arms import TestArms
from robohive.tests.test_fm import TestFM
from robohive.tests.test_hands import TestHMS
from robohive.tests.test_multitask import TestKitchen
from robohive.tests.test_myo import TestMyo
from robohive.tests.test_claws import TestClaws
# from robohive.tests.test_tcdm import TestTCDM

from robohive.tests.test_examine_env import TestExamineEnv
from robohive.tests.test_examine_robot import TestExamineRobot
from robohive.tests.test_logger import TestExamineTrace
from robohive.tests.test_robot import TestRobot

if __name__ == '__main__':
    print("\n=================================", flush=True)
    print("Testing Entire RoboHive")
    unittest.main()