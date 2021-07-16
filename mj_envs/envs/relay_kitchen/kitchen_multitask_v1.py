""" Kitchen environment for long horizon manipulation """

import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mj_envs.robot.robot import Robot

from mujoco_py import MjViewer
import os
from mj_envs.utils.obj_vec_dict import ObsVecDict
import collections
from gym import spaces


OBS_KEYS = ['hand_jnt', 'objs_jnt', 'goal']
# RWD_KEYS = ['reach', 'open', 'bonus']
# RWD_MODE = 'dense' # dense/ sparse

class KitchenV0(mujoco_env.MujocoEnv, utils.EzPickle, ObsVecDict):

    N_DOF_ROBOT = 9

    def __init__(self, *args, **kwargs):
        # get sim
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.sim = mujoco_env.get_sim(model_path=curr_dir+'/assets/franka_kitchen.xml')

        self.robot = Robot( name="relay_kitchen",
                    mj_sim=self.sim,
                    config_path=curr_dir+'/assets/franka_kitchen.config',
                    random_generator=None,
                    is_hardware = False,
                    act_mode="vel")

        # get env
        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)

        self.act_mid = np.zeros(self.sim.model.nu)
        self.act_amp = 2.0 * np.ones(self.sim.model.nu)

        self.obs_dict = {}
        self.rwd_dict = {}
        self.goal = np.zeros((30,))
        mujoco_env.MujocoEnv.__init__(self, sim=self.sim, frame_skip=40)


    # Converted to velocity actuation
    # ROBOTS = {'robot': 'adept_envs.franka.robot.franka_robot:Robot_VelAct'}
    # MODEl = os.path.join(
        # os.path.dirname(__file__),
    #     '../assets/franka_kitchen_jntpos_act_ab.xml')
    # N_DOF_OBJECT = 21

    # def __init__(self, robot_params={}, frame_skip=40):
        # self.goal_concat = True
        # self.obs_dict = {}
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        # self.goal = np.zeros((30,))

        # super().__init__(
        #     self.MODEl,
        #     robot=self.make_robot(
        #         n_jnt=self.N_DOF_ROBOT,  #root+robot_jnts
        #         n_obj=self.N_DOF_OBJECT,
        #         **robot_params),
            # frame_skip=frame_skip,
            # camera_settings=dict(
            #     distance=4.5,
            #     azimuth=-66,
            #     elevation=-65,
        #     ),
        # )

        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = self.sim.model.key_qvel[0].copy()

        act_lower = -1*np.ones((self.N_DOF_ROBOT,))
        act_upper =  1*np.ones((self.N_DOF_ROBOT,))
        self.action_space = spaces.Box(act_lower, act_upper)

        obs_upper = 8. * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -66
        self.viewer.cam.elevation = -65
        self.viewer.cam.distance = 4.5
        self.sim.forward()

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    def step(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        # a = self.act_mid + a * self.act_amp  # mean center and scale
        # self.robot.step(
        #     self, a, step_duration=self.frame_skip * self.model.opt.timestep)

        self.last_ctrl = self.robot.step(ctrl_desired=a, step_duration=self.frame_skip*self.sim.model.opt.timestep,
            realTimeSim=self.mujoco_render_frames, render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'score': score,
            'images': np.asarray(self.render(mode='rgb_array'))
        }
        # self.render()
        return obs, reward_dict['r_total'], done, env_info


    def _get_obs(self):
        self.obs_dict = {}
        self.obs_dict['t'] = np.array([self.sim.data.time])

        sen = self.robot.get_sensors(noise_scale=0)
        # self.robot.sync_sim_state(self.robot.sim, self.sim)
        # self.robot.sensor2sim(sen, self.sim)

        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_jnt'] = self.data.qpos[:self.N_DOF_ROBOT].copy()
        self.obs_dict['objs_jnt'] = self.data.qpos[self.N_DOF_ROBOT:].copy()
        self.obs_dict['goal'] = self.goal.copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, OBS_KEYS)
        return obs

    # def _get_obs(self):
    #     t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
    #         self, robot_noise_ratio=self.robot_noise_ratio)

    #     self.obs_dict = {}
    #     self.obs_dict['t'] = t
    #     self.obs_dict['qp'] = qp
    #     self.obs_dict['qv'] = qv
    #     self.obs_dict['obj_qp'] = obj_qp
    #     self.obs_dict['obj_qv'] = obj_qv
    #     self.obs_dict['goal'] = self.goal
    #     if self.goal_concat:
    #         return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path['env_infos']['score'])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path['env_infos']['rewards']['bonus'][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score))

    def close_env(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs//2,))

    def convert_to_active_observation(self, observation):
        return observation


class KitchenTasksV0(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self):
        super(KitchenTasksV0, self).__init__()

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict['true_reward'] = 0.
        reward_dict['bonus'] = 0.
        reward_dict['r_total'] = 0.
        score = 0.
        return reward_dict, score

    # def render(self, mode='human'):
    #     if mode =='rgb_array':
    #         camera = engine.MovableCamera(self.sim, 1920, 2560)
    #         camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
    #         img = camera.render()
    #         return img
    #     else:
    #         super(KitchenTasksV0, self).render()
