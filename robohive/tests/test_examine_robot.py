import click
import click.testing
import unittest
from robohive.tutorials.examine_robot import main as examine_robot
import os

class TestExamineRobot(unittest.TestCase):
    def test_main(self):
        # Call your function and test its output/assertions
        print("\n=================================", flush=True)
        print("Testing examine robot")
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        runner = click.testing.CliRunner()
        result = runner.invoke(examine_robot, ["--sim_path", curr_dir+"/../envs/arms/franka/assets/franka_reach_v0.xml", \
                                            "--config_path", curr_dir+"/../envs/arms/franka/assets/franka_reach_v0.config", \
                                            "--live_render", "False"])
        print("OUTPUT", result.output.strip(), flush=True)
        print("RESULT", result, flush=True)
        print("EXCEPTION", result.exception, flush=True)

        self.assertEqual(result.exception, None, result.exception)

    # def test_offscreen_rendering(self):
    #     # Call your function and test its output/assertions
    #     print("Testing offscreen rendering")
    #     runner = click.testing.CliRunner()
    #     result = runner.invoke(examine_env, ["--env_name", "door-v1", \
    #                                         "--num_episodes", 1, \
    #                                         "--render", "offscreen",\
    #                                         "--camera_name", "top_cam"])
    #     print(result.output.strip())
    #     self.assertEqual(result.exception, None, result.exception)

    # def no_test_scripted_policy_loading(self):
    #     # Call your function and test its output/assertions
    #     print("Testing scripted policy loading")
    #     runner = click.testing.CliRunner()
    #     result = runner.invoke(examine_env, ["--env_name", "door-v1", \
    #                                         "--num_episodes", 1, \
    #                                         "--render", "offscreen",\
    #                                         "--policy_path", "robohive.utils.examine_env.rand_policy"])
    #     print(result.output.strip())
    #     self.assertEqual(result.exception, None, result.exception)

if __name__ == '__main__':
    print("\n=================================", flush=True)
    print("Testing Examine Robot")
    unittest.main()