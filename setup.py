""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mj_envs',
    version='0.2.0',
    packages=find_packages(),
    description='environments simulated in MuJoCo',
    long_description=read('README.md'),
    url='https://github.com/vikashplus/mj_envs.git',
    author='Movement Control Lab, UW',
    install_requires=[
        'click', 'gym==0.13', 'mujoco-py<2.1,>=2.0', 'termcolor', 'sk-video', 'flatten_dict'
    ],
)
