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
   cd robohive
   python robohive/utils/examine_env.py -e FrankaReachRandom-v0
   ```
   See here for [detailed installation instructions](./setup/README.md) and [frequently asked questions](./setup/FAQ.md).

# Suites
*RoboHive* contains a variety of environement, which are organized as suites. Each suites is a collection of loosely related environements. Following suites are provided at the moment with plans to improve the diversity of the collection.

## 1. Hand Manipulation Suite
HMS contains a collection of environement centered around dexterous manipulation with anthroporphic 24 degrees of freedom  [Adroit Hand](https://vikashplus.github.io/P_Hand.html). These environments were designed for the publication: [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations, RSS2018](https://sites.google.com/corp/view/deeprl-dexterous-manipulation).

Hand-Manipulation-Suite Tasks [(video)](https://youtu.be/jJtBll8l_OM)
![Alt text](robohive/envs/hands/assets/tasks.jpg?raw=false "Adroit Tasks")

## 2. More coming soon
