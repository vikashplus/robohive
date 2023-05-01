import unittest
from robohive.robot.robot import demo_robot

class TestRobot(unittest.TestCase):
    def test_robot(self):
        # Call your function and test its output/assertions
        print("Testing robot")
        demo_robot()

if __name__ == '__main__':
    unittest.main()