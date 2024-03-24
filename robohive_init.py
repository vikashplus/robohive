import os
import shutil
from os.path import expanduser

import git

curr_dir = os.path.dirname(os.path.abspath(__file__))
simhive_path = os.path.join(curr_dir, 'robohive', 'simhive')


# from robohive.utils.import_utils import fetch_git

def fetch_git(repo_url, commit_hash, clone_directory, clone_path=None):
    """
    fetch git repo using provided details
    """
    if clone_path is None:
        clone_path = os.path.join(expanduser("~"), ".robohive")
    clone_directory = os.path.join(clone_path, clone_directory)

    try:
        # Create the clone directory if it doesn't exist
        os.makedirs(clone_directory, exist_ok=True)

        # Clone the repository to the specified path
        if not os.path.exists(os.path.join(clone_directory,'.git')):
            repo = git.Repo.clone_from(repo_url, clone_directory)
            print(f"{repo_url} cloned at {clone_directory}")
        else:
            repo = git.Repo(clone_directory)
            origin = repo.remote('origin')
            origin.fetch()

        # Check out the specific commit if not already
        current_commit_hash = repo.head.commit.hexsha
        if current_commit_hash != commit_hash:
            repo.git.checkout(commit_hash)
            print(f"{repo_url}@{commit_hash} fetched at {clone_directory}")
    except git.GitCommandError as e:
        print(f"Error: {e}")

    return clone_directory


def clean_simhive():
    """
    Remove cached simhive if it exists
    """
    print("RoboHive:> Clearing SimHive ...")
    if os.path.exists(simhive_path):
        shutil.rmtree(simhive_path)
    else:
        print("RoboHive:> SimHive directory does not exist.")
    print("RoboHive:> SimHive cleared")


def fetch_simhive():
    """
    fetch a copy of simhive
    """
    print("RoboHive:> Initializing...")

    # Mark the SimHive version (ToDo: Remove this when commits hashes are auto fetched from submodules)
    __version__ = "0.7.0"

    # Fetch SimHive
    print("RoboHive:> Downloading simulation assets (upto ~300MBs)")
    fetch_git(repo_url="https://github.com/vikashplus/Adroit.git",
                commit_hash="2ef4b752e85782f84fa666fce10de5231cc5c917",
                clone_directory="Adroit",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/furniture_sim.git",
                commit_hash="afb802750acb97f253b2f6fc6e915627d04fcf67",
                clone_directory="furniture_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/franka_sim.git",
                commit_hash="82aaf3bebfa29e00133a6eebc7684e793c668fc1",
                clone_directory="franka_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/scene_sim.git",
                commit_hash="e2eb354dce9aa1797dda081ab2dedf734a8761e6",
                clone_directory="scene_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/sawyer_sim.git",
                commit_hash="affaf56d56be307538e5eed34f647586281762b2",
                clone_directory="sawyer_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/fetch_sim.git",
                commit_hash="7f6d25ae8a6f5778379a48fa60c17d685075e64d",
                clone_directory="fetch_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/dmanus_sim.git",
                commit_hash="b8531308fa34d2bd637d9df468455ae36e2ebcd3",
                clone_directory="dmanus_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/YCB_sim.git",
                commit_hash="46edd9c361061c5d81a82f2511d4fbf76fead569",
                clone_directory="YCB_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/object_sim.git",
                commit_hash="ee0ff14a5369c277687a4636165c5b703bccbf84",
                clone_directory="object_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/trifinger_sim.git",
                commit_hash="49e689ee8d18f5e506ba995aac99822b66700b2b",
                clone_directory="trifinger_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/robel_sim.git",
                commit_hash="4459db96edc359822356030e6223797e257ae4cc",
                clone_directory="robel_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/vikashplus/robotiq_sim.git",
                commit_hash="854d0bfb4e48b076e1d2aa4566c2e23bba17ebae",
                clone_directory="robotiq_sim",
                clone_path=simhive_path)

    fetch_git(repo_url="https://github.com/MyoHub/myo_sim.git",
                commit_hash="5e462da71589fe42164af25ef3c4311231a0d6b2",
                clone_directory="myo_sim",
                clone_path=simhive_path)

    # mark successful creation of simhive
    filename = os.path.join(simhive_path, "simhive-version")
    with open(filename, 'w') as file:
        file.write(__version__)

    print("RoboHive:> Successfully Initialized.")


if __name__ == "__main__":
    fetch_simhive()