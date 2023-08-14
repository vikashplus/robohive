""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym

# RoboHive version
__version__ = "0.6.0"


# Register RoboHive Envs
_current_gym_envs = gym.envs.registration.registry.env_specs.keys()
_current_gym_envs = set(_current_gym_envs)
robohive_env_suite = set()

# Register Arms Suite
import robohive.envs.arms # noqa
robohive_arm_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_arm_suite = set(sorted(robohive_arm_suite))
robohive_env_suite = robohive_env_suite | robohive_arm_suite

# Register Myo Suite
import robohive.envs.myo # noqa
import robohive.envs.myo.myochallenge # noqa
robohive_myo_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_myo_suite
robohive_myo_suite = sorted(robohive_myo_suite)

# Register MyoDex Suite
import robohive.envs.myo # noqa
import robohive.envs.myo.myodex # noqa
robohive_myodex_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_myodex_suite
robohive_myodex_suite = sorted(robohive_myodex_suite)

# Register FM suite
import robohive.envs.fm # noqa
robohive_fm_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_fm_suite
robohive_fm_suite = sorted(robohive_fm_suite)

# Register Hands Suite
import robohive.envs.hands # noqa
# import robohive.envs.tcdm # noqa # WIP
robohive_hand_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_hand_suite
robohive_hand_suite = sorted(robohive_hand_suite)

# Register Claw suite
import robohive.envs.claws # noqa
robohive_claw_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_claw_suite
robohive_claw_suite = sorted(robohive_claw_suite)

# Register Multi-task Suite
import robohive.envs.multi_task # noqa
robohive_multitask_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_multitask_suite
robohive_multitask_suite = sorted(robohive_multitask_suite)

# Register Locomotion Suite
import robohive.envs.quadrupeds # noqa
robohive_quad_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_quad_suite
robohive_quad_suite = sorted(robohive_quad_suite)

# All RoboHive Envs
robohive_env_suite = sorted(robohive_env_suite)