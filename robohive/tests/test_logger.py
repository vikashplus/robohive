import unittest
import click
import click.testing

from robohive.logger.grouped_datasets import test_trace, test_trace_append, test_trace_plot
from robohive.logger.examine_logs import examine_logs
from robohive.utils.examine_env import main as examine_env
import os
import re
import glob

class TestTrace(unittest.TestCase):
    def test_trace(self):
        # Call your function and test its output/assertions
        print("Testing Trace Basics")
        test_trace()

    def test_trace_append(self):
        # Call your function and test its output/assertions
        print("Testing Trace complex appends")
        test_trace_append()

    def test_trace_plot(self):
        # Call your function and test its output/assertions
        print("Testing Trace plotting")
        test_trace_plot()
        # Define the pattern for the files you want to delete
        pattern = "./*plot*.pdf"

        # Use glob to find all files matching the pattern
        files_to_delete = glob.glob(pattern)

        # Iterate over the list of files and delete each one
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


class TestExamineTrace(unittest.TestCase):
    def test_logs_playback(self):

        print("\nTesting logger: Logs playback")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "none",\
                                            "--save_paths", True,\
                                            "--output_name", "door_test_logs"])
        log_name_pattern = re.compile(r'Saved: (?:.+\.h5)')
        log_name = log_name_pattern.search(result.output)[0][7:]

        result = runner.invoke(examine_logs, ["--env_name", "door-v1", \
                                            "--rollout_path", log_name, \
                                            "--render", "none",\
                                            "--mode", "playback"])
        self.assertEqual(result.exception, None, result.exception)

        print(result.output.strip())
        os.remove(log_name)


    def test_logs_render(self):

        print("\nTesting logger: Logs render")
        runner = click.testing.CliRunner()
        result = runner.invoke(examine_env, ["--env_name", "door-v1", \
                                            "--num_episodes", 1, \
                                            "--render", "none",\
                                            "--save_paths", True,\
                                            "--output_name", "door_test_logs"])
        log_name = result.output.strip()[-38:]
        self.assertEqual(result.exception, None, result.exception)


        result = runner.invoke(examine_logs, ["--env_name", "door-v1", \
                                            "--rollout_path", log_name, \
                                            "--render", "none",\
                                            "--mode", "render"])
        print(result.output.strip())
        self.assertEqual(result.exception, None, result.exception)

        os.remove(log_name)


if __name__ == '__main__':
    print("\n=================================", flush=True)
    print("Testing logger")
    unittest.main()