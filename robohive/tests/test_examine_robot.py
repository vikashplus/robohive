import click
import click.testing
import unittest
from robohive.tutorials.examine_robot import main as examine_robot


class TestExamineRobot(unittest.TestCase):
    def test_main(self):
        # Call your function and test its output/assertions
        print("Testing examine robot")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_robot, ["--sim_path", "envs/arms/franka/assets/franka_reach_v0.xml", \
                                            "--config_path", "envs/arms/franka/assets/franka_reach_v0.config", \
                                            "--live_render", "False"])
        print(result.output.strip())
        self.assertEqual(result.exception, None)

    # def test_offscreen_rendering(self):
    #     # Call your function and test its output/assertions
    #     print("Testing offscreen rendering")
    #     runner = click.testing.CliRunner()
    #     result = runner.invoke(examine_env, ["--env_name", "door-v1", \
    #                                         "--num_episodes", 1, \
    #                                         "--render", "offscreen",\
    #                                         "--camera_name", "top_cam"])
    #     print(result.output.strip())
    #     self.assertEqual(result.exception, None)

    # def no_test_scripted_policy_loading(self):
    #     # Call your function and test its output/assertions
    #     print("Testing scripted policy loading")
    #     runner = click.testing.CliRunner()
    #     result = runner.invoke(examine_env, ["--env_name", "door-v1", \
    #                                         "--num_episodes", 1, \
    #                                         "--render", "offscreen",\
    #                                         "--policy_path", "robohive.utils.examine_env.rand_policy"])
    #     print(result.output.strip())
    #     self.assertEqual(result.exception, None)

if __name__ == '__main__':
    unittest.main()