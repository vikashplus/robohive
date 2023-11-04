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


class StandBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'root_pos',
        'root_euler',
        'kitty_qpos',
        'root_vel',
        'root_angular_vel',
        'kitty_qvel',
        'last_action',
        'upright',
        'pose_error',
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        'pose_error_cost': 4.0,
        'center_distance_cost': 2.0,
        'bonus_small': 5.0,
        'bonus_big': 10.0,
        'upright': 1.0,
        'falling': 100,
        }

    DEFAULT_VISUAL_KEYS = [
        'rgb:A:headCam:256x256:2d',
        'rgb:A:tailCam:256x256:2d',
        'rgb:A:leftCam:256x256:2d',
        'rgb:A:rightCam:256x256:2d',
        ]

    DEFAULT_PROPRIO_KEYS = [
        "kitty_qpos",   # radian
    ]

    ENV_CREDIT = """\
    ROBEL: RObotics BEnchmarks for Learning with low-cost robots
        Michael Ahn, Henry Zhu, Kristian Hartikainen, Hugo Ponte
        Abhishek Gupta, Sergey Levine, Vikash Kumar
        CoRL-2019 | https://sites.google.com/view/roboticsbenchmarks/
    """

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.ENV_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
                dof_range_names=None,
                act_range_names=None,
                upright_threshold: float = 0.0,
                torso_site_name='torso',
                target_site_name='target',
                heading_site_name='heading',
                target_distance_range = (2.3, 2.3),
                target_angle_range = (np.pi/2, np.pi/2),
                target_height_range = (0.28, 0.28),
                frame_skip = 40,
                obs_keys=DEFAULT_OBS_KEYS,
                weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
                reset_type='fixed',
                proprio_keys=DEFAULT_PROPRIO_KEYS,
                **kwargs,
        ):

        self._upright_threshold = upright_threshold
        self.reset_type = reset_type

        # ids
        self.torso_sid = self.sim.model.site_name2id(torso_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.heading_sid = self.sim.model.site_name2id(heading_site_name)
        self.target_distance_range, self.target_angle_range, self.target_height_range = target_distance_range, target_angle_range, target_height_range

        self.dof_range = range(self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(dof_range_names[0])],
                               self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(dof_range_names[1])]+1)
        self.act_range = range(self.sim.model.actuator_name2id(act_range_names[0]),
                               self.sim.model.actuator_name2id(act_range_names[1])+1)
        self._desired_pose = np.zeros(len(self.dof_range))

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       proprio_keys=proprio_keys,
                       **kwargs)


    def get_obs_dict(self, sim):
        target_xy = sim.data.site_xpos[self.target_sid, :2]
        heading_xy = sim.data.site_xpos[self.heading_sid, :2]
        kitty_xy = sim.data.site_xpos[self.torso_sid, :2]
        kitty_xyz = sim.data.site_xpos[self.torso_sid, :3]

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
        target_error_rel = np.matmul(rot_mat[:2,:2].transpose(), (target_xy - kitty_xy))

        obs_dict = collections.OrderedDict((
            # Add observation terms relating to being upright.
            ('time', np.array([self.time])),
            ('last_action', self.last_ctrl.copy()),
            ('upright', np.array([up])),
            ('root_pos', kitty_xyz.copy()), # sim.data.qpos[:3], torso_track_state.pos),
            ('root_euler', sim.data.qpos[3:6].copy()), #torso_track_state.rot_euler),
            ('root_vel', sim.data.qvel[:3].copy()), # torso_track_state.vel),
            ('root_angular_vel', sim.data.qvel[3:6].copy()), #torso_track_state.angular_vel),
            ('kitty_qpos', sim.data.qpos[self.dof_range].copy()), #robot_state.qpos),
            ('kitty_qvel', sim.data.qvel[self.dof_range].copy()), #robot_state.qvel),
            ('heading', np.array([heading])),
            ('target_pos', target_xy),
            ('target_error', target_error_rel),
            ('pose_error', self._desired_pose - sim.data.qpos[self.dof_range]),
        ))
        return obs_dict


    def get_reward_dict(self, obs_dict):
        """Returns the reward for the given action and observation."""

        pose_mean_error = np.abs(obs_dict['pose_error']).mean(axis=-1)
        upright = obs_dict['upright'][:,:,0] if obs_dict['upright'].ndim==3 else obs_dict['upright'][0]
        center_dist = np.linalg.norm(obs_dict['root_pos'][:2], axis=-1)

        rwd_dict = collections.OrderedDict((
            # staying upright
            ('upright', (upright - self._upright_threshold)/(1 - self._upright_threshold)),
            # not falling
            ('falling', -1.0* (upright < self._upright_threshold)),
            # Reward for closeness to desired pose.
            ('pose_error_cost', -1 * pose_mean_error),
            # Reward for closeness to center; i.e. being stationary.
            ('center_distance_cost', -1 * center_dist),
            # Bonus when mean error < 30deg, scaled by uprightedness.
            ('bonus_small', 1.0 * (pose_mean_error < (np.pi / 6)) * upright),
            # Bonus when mean error < 15deg and upright within 30deg.
            ('bonus_big', 1.0 * (pose_mean_error < (np.pi / 12)) * (upright > 0.9)),
            # Must keys
            ('sparse', (1 - np.maximum(pose_mean_error / (np.pi / 3), 1))),  # Normalized pose error by 60deg.
            ('solved', (pose_mean_error < (np.pi / 12)) * (upright > 0.9)),
            ('done', upright < self._upright_threshold),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def reset(self, reset_qpos=None, reset_qvel=None):

        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            quad_pose = np.zeros(len(self.dof_range))

            if self.reset_type.lower() == 'fixed':
                quad_pose[[0, 3, 6, 9]] = 0
                quad_pose[[1, 4, 7, 10]] = np.pi / 4
                quad_pose[[2, 5, 8, 11]] = -np.pi / 2
                reset_qpos[self.dof_range] = quad_pose

            elif self.reset_type.lower() == 'random':
                for name, device in self.robot.robot_config.items():
                    for act_id, actuator in enumerate(device['actuator']):
                        reset_qpos[actuator['data_id']] = self.np_random.uniform(low=actuator['pos_range'][0], high=actuator['pos_range'][1])
            else:
                raise TypeError(f"Unknown reset type: {self.reset_type}")

        obs = super().reset(reset_qpos, reset_qvel)
        return obs

