from robohive.utils.import_utils import fetch_git
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
simhive_path = os.path.join(curr_dir, 'robohive', 'simhive')

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
            commit_hash="58d561fa416b6a151761ced18f2dc8f067188909",
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
            commit_hash="87cd8dd5a11518b94fca16bc22bb04f6836c6aa7",
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
            commit_hash="1758d66ad066542e87ed2fa4ffc69121195aba2c",
            clone_directory="myo_sim",
            clone_path=simhive_path)