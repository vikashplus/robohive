""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """
import argparse
import os
import sys
from typing import List

from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print(
        "This library is only compatible with Python 3, but you are running "
        "Python {}. The installation will likely fail.".format(
            sys.version_info.major
            )
        )


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mj_envs setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="mj_envs",
        help="the name of this output wheel",
    )
    return parser.parse_known_args(argv)


def _main(argv):
    args, unknown = parse_args(argv)
    sys.argv = [sys.argv[0], *unknown]
    name = args.package_name

    extra_files = package_files('mj_envs')

    setup(
        name=name,
        version='0.4.0',
        packages=find_packages(),
        package_data={"": extra_files},
        include_package_data=True,
        description='environments simulated in MuJoCo',
        long_description=read('README.md'),
        long_description_content_type="text/markdown",
        url='https://github.com/vikashplus/mj_envs.git',
        author='Movement Control Lab, UW',
        license='Apache 2.0',
        install_requires=[
            'click',
            'gym==0.13',
            'free-mujoco-py',
            'numpy==1.22.4',
            'termcolor',
            'sk-video',
            'flatten_dict',
            'matplotlib',
            'ffmpeg',
            'absl-py',
            # 'r3m @ git+https://github.com/facebookresearch/r3m.git',
            # 'data_tools @ git+https://github.com/fairinternal/data_tools.git',
            'h5py==3.7.0',
        ],
        extras_require={
            'a0': [
                'pycapnp==1.1.0',
                'alephzero',  # real_sense subscribers dependency
            ]
        }
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
