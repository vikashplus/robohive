import click
import click.testing
import unittest
from robohive.utils.examine_env import main as examine_env
import os


class TestExamineEnv(unittest.TestCase):
    def test_main(self):
        # Call your function and test its output/assertions
        print("Testing env with random policy")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "none"])
        print(result.output.strip())
        self.assertEqual(result.exception, None)

    def test_offscreen_rendering(self):
        # Call your function and test its output/assertions
        print("Testing offscreen rendering")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "offscreen",\
                                            "--camera_name", "view_1"])
        print(result.output.strip())
        self.assertEqual(result.exception, None)
        os.remove('random_policy0.mp4')

    def no_test_scripted_policy_loading(self):
        # Call your function and test its output/assertions
        print("Testing scripted policy loading")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "offscreen",\
                                            "--policy_path", "robohive.utils.examine_env.rand_policy"])
        print(result.output.strip())
        self.assertEqual(result.exception, None)

if __name__ == '__main__':
    print("\n=================================", flush=True)
    print("Testing Examine Env")
    unittest.main()