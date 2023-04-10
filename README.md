<!-- =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= -->

# RoboHive
![PyPI](https://img.shields.io/pypi/v/robohive)
![PyPI - License](https://img.shields.io/pypi/l/robohive)
[![Downloads](https://pepy.tech/badge/robohive)](https://pepy.tech/project/robohive)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rdSgnsfUaE-eFLjAkFHeqfUWzAK8ruTs?usp=sharing)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://robohiveworkspace.slack.com)

`RoboHive` is a collection of environments/tasks simulated with the [MuJoCo](http://www.mujoco.org/) physics engine exposed using the OpenAI-Gym API.

## Getting Started
   Getting started with RoboHive is as simple as -
   ``` bash
   # Install RoboHive and demo an environemnt
   pip install robohive
   python -m robohive.utils.examine_env -e FrankaReachRandom-v0
   ```

   or, alternatively,

   ``` bash
   git clone --recursive https://github.com/vikashplus/robohive.git; cd robohive
   pip install -e .
   python -m robohive.utils.examine_env -e FrankaReachRandom-v0
   ```


   See here for [detailed installation instructions](./setup/README.md) and [frequently asked questions](https://github.com/vikashplus/robohive/wiki/6.-Tutorials-&-FAQs#installation).

# Suites
*RoboHive* contains a variety of environement, which are organized as suites. Each suites is a collection of loosely related environements. Following suites are provided at the moment with plans to improve the diversity of the collection.

## - Hand Manipulation Suite

   This suite contains a collection of environement centered around dexterous manipulation. Standard ADROIT benchmarks introduced in [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations, RSS2018](https://sites.google.com/corp/view/deeprl-dexterous-manipulation).) are a part of this suite
## - Arm Manipulation Suite

   This suite contains a collection of environement centered around Arm+Gripper manipulation.

## - Myo Suite

   This suite contains a collection of environements related to biomechanics. Standard [MyoSuite benchmarks](https://sites.google.com/view/myosuite) are a part of this suite

## - MultiTask Suite

   This suite contains a collection of environement centered around multi-tassk. Standard [RelayKitchen benchmarks](https://relay-policy-learning.github.io/) are a part of this suite
## - TCDM Suite (WIP)
   This suite contains a collection of environement centered around dexterous manipulation. Standard [TCDM benchmarks](https://pregrasps.github.io/) are a part of this suite

## - ROBEL Suite (Coming soon)
   This suite contains a collection of environement centered around real world locomotion and manipulation. Standard [ROBEL benchmarks](http://roboticsbenchmarks.org/) are a part of this suite
