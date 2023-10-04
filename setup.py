""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This library is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

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
    version='0.6.0',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={"": extra_files},
    include_package_data=True,
    description='environments simulated in MuJoCo',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/vikashplus/robohive.git',
    author='Movement Control Lab, UW',
    install_requires=[
        'click',
        'gym==0.13',
        'mujoco==2.3.7',
        'dm-control==1.0.14',
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
