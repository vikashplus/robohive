""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym

_current_gym_envs = gym.envs.registration.registry.env_specs.keys()
_current_gym_envs = set(_current_gym_envs)

import robohive.envs.arms # noqa
import robohive.envs.myo # noqa
import robohive.envs.myo.myochallenge # noqa
import robohive.envs.fm # noqa
import robohive.envs.hands # noqa
import robohive.envs.multi_task # noqa
# import robohive.envs.tcdm # noqa # WIP
import robohive.envs.claws # noqa
import robohive.envs.quadrupeds # noqa

robohive_envs = set(gym.envs.registration.registry.env_specs.keys()) - _current_gym_envs
robohive_envs = sorted(robohive_envs)

__version__ = "0.5.0"
