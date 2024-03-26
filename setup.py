""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
import shutil
import sys

from setuptools import find_packages, setup

# Check and warn if FFmpeg is not available
if shutil.which("ffmpeg") is None:
    help = """FFmpeg not found in your system. Please install FFmpeg before proceeding
          Options:
            (1) LINUX: apt-get install ffmpeg
            (2) OSX: brew install ffmpeg"""
    raise ModuleNotFoundError(help)

if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 8):
    print("This library requires Python 3.8 or higher, but you are running "
          "Python {}.{}. The installation will likely fail.".format(sys.version_info.major, sys.version_info.minor))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('robohive')

setup(
    name='robohive',
    version='0.7.0',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={"": extra_files+['../robohive_init.py']},
    include_package_data=True,
    description='A Unified Framework for Robot Learning',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/vikashplus/robohive.git',
    author='Vikash Kumar',
    author_email="vikahsplus@gmail.com",
    install_requires=[
        'click',
        # 'gym==0.13',  # default to this stable point if caught in gym issues.
        'gymnasium==0.29.1',
        'mujoco==3.1.3',
        'dm-control==1.0.16',
        'termcolor',
        'sk-video',
        'flatten_dict',
        'matplotlib',
        'ffmpeg',
        'absl-py',
        'torch',
        'h5py==3.7.0',
        'pink-noise-rl',
        'gitpython'
    ],
    extras_require={
      # To use mujoco bindings, run (pip install -e ".[mujoco]") and set sim_backend=MUJOCO
      'mujoco_py':[
        'free-mujoco-py',
        ],
      # To use hardware dependencies, run (pip install -e ".[a0]") and follow install instructions inside robot
      'a0': [
        'pycapnp>=1.1.1',
        'alephzero', # real_sense subscribers dependency
        ],
      'encoder':[
          'torchvision',
        # Unlike pypi, Git dependencies can be directly installed in editable mode.
        # To use r3m/vc encoders, uncomment below and run (pip install -e ".[encoder]")
        # 'r3m @ git+https://github.com/facebookresearch/r3m.git',
        # 'vc_models @ git+https://github.com/facebookresearch/eai-vc.git@9958b278666bcbde193d665cc0df9ccddcdb8a5a#egg=vc_models&subdirectory=vc_models',
      ]
    },
    entry_points={
        'console_scripts': [
            'robohive_init = robohive_init:fetch_simhive',
            'robohive_clean = robohive_init:clean_simhive',
        ],
    },
)
