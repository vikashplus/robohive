<!-- =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= -->

# Mujoco Environments
`mj_envs` is a collection of environments/tasks simulated with the [Mujoco](http://www.mujoco.org/) physics engine and wrapped in the OpenAI `gym` API.

## Getting Started
`mj_envs` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

0. We recommend installaition within a conda environement. If you don't have one yet, create one using
```
conda create -n robohive python=3
```

1. Clone this repo on branch `branch_name` with pre-populated submodule dependencies

a. Most users -
```
git clone -c submodule.mj_envs/sims/neuromuscular_sim.update=none --branch <branch_name> --recursive https://github.com/vikashplus/mj_envs.git
```
b. myoSuite developers: you must have access to neuromuscular_sim(private repo) -
```
git clone --branch <branch_name> --recursive https://github.com/vikashplus/mj_envs.git
```
2. Install package using `pip`
```
$ cd mj_envs
$ pip install -e .
```
**OR**
Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
```
export PYTHONPATH="<path/to/mj_envs>:$PYTHONPATH"
```
3. You can visualize the environments with random controls using the below command
```
$ python mj_envs/utils/examine_env.py -e FrankaReachRandom-v0
```
**FAQ:**
1. If the visualization results in a GLFW error, this is because `mujoco-py` does not see some graphics drivers correctly. This can usually be fixed by explicitly loading the correct drivers before running the python script. See [this page](https://github.com/aravindr93/mjrl/tree/master/setup#known-issues) for details.
2. If FFmpeg isn't found then run `apt-get install ffmpeg` on linux and `brew install ffmpeg` on osx (`conda install FFmpeg` causes some issues)


# modules
*mj_envs* contains a variety of environements, which are organized as modules. Each module is a collection of loosely related environements. Following modules are provided at the moment with plans to improve the diversity of the collection.

## 1. Hand Manipulation Suite (HMS)
HMS contains a collection of environements centered around dexterous manipulation with anthroporphic 24 degrees of freedom  [Adroit Hand](https://vikashplus.github.io/P_Hand.html). These environments were designed for the publication: [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations, RSS2018](https://sites.google.com/corp/view/deeprl-dexterous-manipulation).

Hand-Manipulation-Suite Tasks [(video)](https://youtu.be/jJtBll8l_OM)
:-------------------------:
![Alt text](mj_envs/envs/hand_manipulation_suite/assets/tasks.jpg?raw=false "Adroit Tasks")

## 2. More coming soon
