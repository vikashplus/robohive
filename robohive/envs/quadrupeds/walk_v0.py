""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from robohive.envs import env_base
from robohive.utils.vector_math import calculate_cosine

class WalkBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'dk_upright', 'dk_kitty_qpos', 'dk_heading', 'dk_target_error', 'last_a'
    ]

    # DEFAULT_OBS_KEYS = ['dk_upright', 'dk_root_pos', 'dk_root_euler', 'dk_kitty_qpos', 'dk_heading', 'dk_target_pos', 'dk_target_error', 'last_a']

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
                'dk_upright': 1.0,
                'dk_falling': -5000.0,
                'dk_target_dist_cost': 1.0,
                'dk_heading': 1.0,
                'dk_height': -25.0,
                'dk_bonus_small': 1.0,
                'dk_bonus_big': 1.0
                }


    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)


    def _setup(self,
                dof_range_names=None,
                act_range_names=None,
                upright_threshold: float = 0.9,
                torso_site_name='torso',
                target_site_name='target',
                heading_site_name='heading',
                target_distance_range = (2.3, 2.3),
                target_angle_range = (0., 0.),
                frame_skip = 40,
                obs_keys=DEFAULT_OBS_KEYS,
                weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
                **kwargs,
        ):

        self._upright_threshold = upright_threshold
        # ids
        self.torso_sid = self.sim.model.site_name2id(torso_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.heading_sid = self.sim.model.site_name2id(heading_site_name)
        self.target_distance_range, self.target_angle_range = target_distance_range, target_angle_range
        # self.target_xyz_range = target_xyz_range
        self.dof_range = range(self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(dof_range_names[0])],
                               self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(dof_range_names[1])]+1)
        self.act_range = range(self.sim.model.actuator_name2id(act_range_names[0]),
                               self.sim.model.actuator_name2id(act_range_names[1])+1)

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       **kwargs)


    def get_obs_dict(self, sim):
        target_xy = sim.data.site_xpos[self.target_sid, :2]
        heading_xy = sim.data.site_xpos[self.heading_sid, :2]
        kitty_xy = sim.data.site_xpos[self.torso_sid, :2]

        # Get the heading of the torso (the y-axis).
        current_heading = sim.data.site_xmat[self.torso_sid, [1, 4]]

        # Get the direction towards the heading location.
        desired_heading = heading_xy - kitty_xy

        # Calculate the alignment of the heading with the desired direction.
        heading = calculate_cosine(current_heading, desired_heading)

        # upright
        up = sim.data.site_xmat[self.torso_sid][8]

        # target error in torso coordinate
        rot_mat = np.reshape(sim.data.site_xmat[self.torso_sid], [3, 3])
        dk_target_error_rel = np.matmul(rot_mat[:2,:2].transpose(), (target_xy - kitty_xy))

        obs_dict = collections.OrderedDict((
            # Add observation terms relating to being upright.
            ('time', np.array([self.time])),
            ('last_a', self.last_ctrl.copy()),
            ('dk_upright', np.array([up])),
            ('dk_root_pos', sim.data.qpos[:3].copy()), #torso_track_state.pos),
            ('dk_root_euler', sim.data.qpos[3:6].copy()), #torso_track_state.rot_euler),
            ('dk_root_vel', sim.data.qvel[:3].copy()), # torso_track_state.vel),
            ('dk_root_angular_vel', sim.data.qvel[3:6].copy()), #torso_track_state.angular_vel),
            ('dk_kitty_qpos', sim.data.qpos[self.dof_range].copy()), #robot_state.qpos),
            ('dk_kitty_qvel', sim.data.qvel[self.dof_range].copy()), #robot_state.qvel),
            ('dk_heading', np.array([heading])),
            ('dk_target_pos', target_xy),
            ('dk_target_error', dk_target_error_rel),
        ))
        return obs_dict


    def get_reward_dict(self, obs_dict):
        """Returns the reward for the given action and observation."""
        target_xy_dist = np.linalg.norm(obs_dict['dk_target_error'], axis=-1)
        heading = obs_dict['dk_heading'][:,:,0]
        upright = obs_dict['dk_upright'][:,:,0]

        rwd_dict = collections.OrderedDict((
            # Reward for proximity to the target.
            ('dk_target_dist_cost', -4.0 * target_xy_dist),
            # staying upright
            ('dk_upright', (upright - self._upright_threshold)),
            # not falling
            ('dk_falling', 1.0* (upright < self._upright_threshold)),
            # Heading - 1 @ cos(0) to 0 @ cos(25deg).
            ('dk_heading', 2.0 * (heading - 0.9) / 0.1),
            # height
            ('dk_height', abs(obs_dict['dk_root_pos'][:,:,2]+0.035)),
            # Bonus
            ('dk_bonus_small', 5.0*(target_xy_dist < 0.5) + 5.0*(heading > 0.9)),
            ('dk_bonus_big', 10.0 * (target_xy_dist < 0.5) * (heading > 0.9)),
            # Must keys
            ('sparse',  -1.0*target_xy_dist),
            ('solved',  (target_xy_dist < 0.5)),
            ('done',    obs_dict['dk_upright'][:,:,0] < self._upright_threshold),
        ))

        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def reset(self, reset_qpos=None, reset_qvel=None):
        target_dist = self.np_random.uniform(*self.target_distance_range)
        target_theta = np.pi / 2 + self.np_random.uniform(*self.target_angle_range)

        self.sim.model.site_pos[self.target_sid] = target_dist * np.array([np.cos(target_theta), np.sin(target_theta), 0])
        self.sim.model.site_pos[self.heading_sid] = target_dist * np.array([np.cos(target_theta), np.sin(target_theta), 0])
        obs = super().reset(reset_qpos, reset_qvel)
        return obs

