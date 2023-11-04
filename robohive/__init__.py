""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """


# RoboHive version
__version__ = "0.6.0"


# Check for RoboHive initialization
from robohive.utils.import_utils import simhive_isavailable
simhive_isavailable(robohive_version = __version__)


# Register RoboHive Envs
import gym
_current_gym_envs = gym.envs.registration.registry.env_specs.keys()
_current_gym_envs = set(_current_gym_envs)
robohive_env_suite = set()

# Register Arms Suite
import robohive.envs.arms # noqa
robohive_arm_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_arm_suite = set(sorted(robohive_arm_suite))
robohive_env_suite = robohive_env_suite | robohive_arm_suite

# Register MyoBase Suite
import robohive.envs.myo.myobase # noqa
robohive_myobase_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_myobase_suite
robohive_myobase_suite = sorted(robohive_myobase_suite)

# Register MyoChal Suite
import robohive.envs.myo.myochallenge # noqa
robohive_myochal_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_myochal_suite
robohive_myochal_suite = sorted(robohive_myochal_suite)

# Register MyoDM Suite
import robohive.envs.myo.myodm # noqa
robohive_myodm_suite = set(gym.envs.registration.registry.env_specs.keys())-robohive_env_suite-_current_gym_envs
robohive_env_suite  = robohive_env_suite | robohive_myodm_suite
robohive_myodm_suite = sorted(robohive_myodm_suite)

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