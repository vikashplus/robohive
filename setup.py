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
    version='0.1.1',
    packages=find_packages(),
    description='environments simulated in MuJoCo',
    long_description=read('README.md'),
    url='https://github.com/MovementControl/mj_envs.git',
    author='Movement Control Lab, UW',
    install_requires=[
        'click', 'gym==0.9.3', 'mujoco_py==1.50.1.35', 'termcolor',
    ],
)