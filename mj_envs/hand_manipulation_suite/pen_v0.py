import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mj_envs.utils.quatmath import quat2euler, euler2quat
from mujoco_py import MjViewer
import os

ADD_BONUS_REWARDS = True

class PenEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_pen.xml', 10)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.pen_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
            starting_up = False
        except:
            a = a                             # only for the initialization phase
            starting_up = True
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs, a)
        done = False if starting_up else self.get_done(obs)
        return obs, reward, done, self.get_env_infos()

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        return np.concatenate([qp[:-6], self.data.qvel.ravel() * self.dt, obj_orien, desired_orien, obj_pos, obj_pos-desired_pos])
        
        # return np.concatenate([obs_dict['qp'][:-6], obs_dict['obj_pos'], obs_dict['obj_vel'], obs_dict['obj_orien']
        #         , obs_dict['desired_orien'], obs_dict['obj_pos']-obs_dict['desired_pos'], obs_dict['obj_orien']-
        #         obs_dict['desired_orien']])


    def get_reward(self, obs, act):
        obs = np.clip(obs, -10.0, 10.0)
        act = np.clip(act, -1.0, 1.0)
        if len(obs.shape) == 1:
            obj_pos = obs[-6:-3]
            obj_pos_delta = obs[-3:]
            obj_orien = obs[-12:-9]
            des_orien = obs[-9:-6]
            # pos cost
            obj_dist = np.linalg.norm(obj_pos_delta)
            # orientation cost
            orien_similarity = np.sum(obj_orien * des_orien)
            # pen dropped
            pen_dropped = 5.0 if obj_pos[2] < 0.075 else 0.0
        else:
            obj_pos = obs[:, :, -6:-3]
            obj_pos_delta = obs[:, :, -3:]
            obj_orien = obs[:, :, -12:-9]
            des_orien = obs[:, :, -9:-6]
            # pos cost
            obj_dist = np.linalg.norm(obj_pos_delta, axis=-1)
            # orientation cost
            orien_similarity = np.sum(obj_orien * des_orien, axis=-1)
            orien_similarity = np.clip(orien_similarity, -1.0, 1.0)
            # pen dropped
            pen_dropped = 5.0 * (obj_pos[:, :, 2] < 0.075)
        reward = orien_similarity - obj_dist - pen_dropped
        if ADD_BONUS_REWARDS:
            reward = reward + 10.0 * (obj_dist < 0.075) * (orien_similarity > 0.9) + \
                              50.0 * (obj_dist < 0.075) * (orien_similarity > 0.95)
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        act = paths["actions"]
        rewards = self.get_reward(obs, act)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    def get_done(self, obs):
        obj_pos = obs[-6:-3]
        done = True if obj_pos[2] < 0.075 else False
        return done

    def truncate_paths(self, paths):
        for path in paths:
            obs = path["observations"]
            obj_height = obs[:, -4]
            T = obs.shape[0]
            t = 0
            done = False
            while t < T and done is False:
                done = True if obj_height[t] < 0.075 else False
                t = t + 1
                T = t if done else T
            path["observations"] = path["observations"][:T]
            path["actions"] = path["actions"][:T]
            path["rewards"] = path["rewards"][:T]
            path["terminated"] = done
        return paths

    # def step(self, a):
    #     a = np.clip(a, -1.0, 1.0)
    #     try:
    #         starting_up = False
    #         a = self.act_mid + a*self.act_rng # mean center and scale
    #     except:
    #         starting_up = True
    #         a = a                             # only for the initialization phase
    #     self.do_simulation(a, self.frame_skip)

    #     obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
    #     desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
    #     obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
    #     desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length

    #     # pos cost
    #     dist = np.linalg.norm(obj_pos-desired_loc)
    #     reward = -dist
    #     # orien cost
    #     orien_similarity = np.dot(obj_orien, desired_orien)
    #     reward += orien_similarity

    #     if ADD_BONUS_REWARDS:
    #         # bonus for being close to desired orientation
    #         if dist < 0.075 and orien_similarity > 0.9:
    #             reward += 10
    #         if dist < 0.075 and orien_similarity > 0.95:
    #             reward += 50

    #     # penalty for dropping the pen
    #     done = False
    #     if obj_pos[2] < 0.075:
    #         reward -= 5
    #         done = True if not starting_up else False

    #     goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

    #     return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()

    def get_env_infos(self):
        state = self.get_env_state()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        dist = np.linalg.norm(obj_pos-desired_loc)
        orien_similarity = np.dot(obj_orien, desired_orien)
        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False
        return dict(state=state, goal_achieved=goal_achieved)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 5 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 5:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
