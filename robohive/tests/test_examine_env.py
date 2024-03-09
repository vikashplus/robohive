import click
import click.testing
import unittest
from robohive.utils.examine_env import main as examine_env
import os
import glob
import time


class TestExamineEnv(unittest.TestCase):

    def delete_recent_file(self, filename_pattern, directory='.', age=5):

        # Get the current time
        current_time = time.time()

        # Use glob to find files matching the pattern in the specified directory
        matching_files = glob.glob(os.path.join(directory, filename_pattern))

        # Iterate over the matching files
        for file_path in matching_files:
            try:
                # Get the creation time of the file
                creation_time = os.path.getctime(file_path)

                # Calculate the time difference between current time and creation time
                time_difference = current_time - creation_time

                # If the file was created within the last 5 seconds, delete it
                if time_difference <= 5:
                    os.remove(file_path)
                    print(f"Deleted file created within {age} seconds: {file_path}")
                else:
                    print(f"File not deleted: {file_path}, created {time_difference} seconds ago.")
            except Exception as e:
                print(f"Error deleting file: {file_path} - {e}")


    def test_main(self):
        # Call your function and test its output/assertions
        print("Testing env with random policy")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "none"])
        print("OUTPUT", result.output.strip(), flush=True)
        print("RESULT", result, flush=True)
        print("EXCEPTION", result.exception, flush=True)
        # print(result.output.strip())
        self.assertEqual(result.exception, None, result.exception)

    def test_offscreen_rendering(self):
        # Call your function and test its output/assertions
        print("Testing offscreen rendering")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "offscreen",\
                                            "--camera_name", "view_1"])
        print("OUTPUT", result.output.strip(), flush=True)
        print("RESULT", result, flush=True)
        print("EXCEPTION", result.exception, flush=True)
        # print(result.output.strip())
        self.assertEqual(result.exception, None, result.exception)
        self.delete_recent_file(filename_pattern="random_policy*.mp4")

    def test_paths_plotting(self):
        # Call your function and test its output/assertions
        print("Testing plotting paths")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "none",\
                                            "--plot_paths", True])
        print("OUTPUT", result.output.strip(), flush=True)
        print("RESULT", result, flush=True)
        print("EXCEPTION", result.exception, flush=True)
        # print(result.output.strip())
        self.assertEqual(result.exception, None, result.exception)
        self.delete_recent_file(filename_pattern="random_policy*Trial*.pdf")

    def no_test_scripted_policy_loading(self):
        # Call your function and test its output/assertions
        print("Testing scripted policy loading")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "offscreen",\
                                            "--policy_path", "robohive.utils.examine_env.rand_policy"])
        print("OUTPUT", result.output.strip(), flush=True)
        print("RESULT", result, flush=True)
        print("EXCEPTION", result.exception, flush=True)
        # print(result.output.strip())
        self.assertEqual(result.exception, None, result.exception)

if __name__ == '__main__':
    print("\n=================================", flush=True)
    print("Testing Examine Env")
    unittest.main()