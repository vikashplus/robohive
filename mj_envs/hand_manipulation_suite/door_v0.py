import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os
from darwin.darwin_utils.obs_vec_dict import ObsVecDict
import collections

# NOTES:
#     1. why is qpos[0] not a part of the obs?

OBS_KEYS = ['hand_jnt', 'latch_pos', 'door_pos', 'palm_pos', 'handle_pos', 'reach_err', 'door_open']
# OBS_KEYS = ['hand_jnt', 'latch_pos', 'door_pos', 'palm_pos', 'handle_pos', 'reach_err']
# RWD_KEYS = ['reach', 'open', 'smooth', 'bonus']
RWD_KEYS = ['reach', 'open', 'bonus']

ADD_BONUS_REWARD = True

class DoorEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, ObsVecDict):
    def __init__(self):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)
        self.obs_dict = {}
        self.rwd_dict = {}
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_door.xml', 5)
        
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        # ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        
        obs = self.get_obs()
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # reward = self.get_reward(obs, a)
        # reward = self.get_reward_old()
        
        # reward_dict = self.get_reward_dict(self.obs_dict)

        # print(reward-reward_dict['total'][0][0])

        # finalize step
        env_info = self.get_env_infos()
        env_info['done'] = False
        # print(env_info['done'])
        # import ipdb; ipdb.set_trace()

        return obs, env_info['reward'], env_info['done'], env_info
        # return obs, env_info['reward'], env_info['done'], env_info



        # return obs, reward, env_info['done'], env_info
        # return obs, env_info['rewards'], False, env_info
        # print(env_info['done'])

        # done = False
        # return obs, reward, done, env_info
        # goal_achieved = True if door_pos >= 1.35 else False
        # return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        self.obs_dict['t'] = np.array([self.sim.data.time])

        self.obs_dict['hand_jnt'] = self.data.qpos[1:-2].copy()
        self.obs_dict['hand_vel'] = self.data.qvel[:-2].copy()
        self.obs_dict['handle_pos'] = self.data.site_xpos[self.handle_sid].copy()
        self.obs_dict['palm_pos'] = self.data.site_xpos[self.grasp_sid].copy()
        self.obs_dict['reach_err'] = self.obs_dict['palm_pos']-self.obs_dict['handle_pos']
        self.obs_dict['door_pos'] = np.array([self.data.qpos[self.door_hinge_did]])
        self.obs_dict['latch_pos'] = np.array([self.data.qpos[-1]])
        self.obs_dict['door_open'] = 2.0*(self.obs_dict['door_pos'] > 1.0) -1.0
        
        obs = self.obsdict2obsvec(self.obs_dict, OBS_KEYS)
        return obs

    def get_reward_old(self):
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARD:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10
        return reward
        
    def get_reward(self, obs, act):
        obs = np.clip(obs, -10.0, 10.0)
        act = np.clip(act, -1.0, 1.0)
        
        if len(obs.shape) == 1:
            door_pos = obs[28]
            palm_pos = obs[29:32]
            handle_pos = obs[32:35]
            # get to handle
            reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
            # open door
            reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
            reward += -1e-5*np.sum(self.data.qvel**2)
        else:
            door_pos = obs[:, :, 28]
            palm_pos = obs[:, :, 29:32]
            handle_pos = obs[:, :, 32:35]
            # get to handle
            reward = -0.1*np.linalg.norm(palm_pos-handle_pos, axis=-1)
            # open door
            reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        
        # Bonus
        if ADD_BONUS_REWARD:
            reward += 2*(door_pos > 0.2) + 8*(door_pos > 1.0) + 10*(door_pos > 1.35)

        return reward

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['palm_pos']-obs_dict['handle_pos'], axis=-1)
        door_pos = obs_dict['door_pos'][:,:,0]
        self.rwd_dict = collections.OrderedDict((
            ('reach', -0.1* reach_dist),
            ('open', -0.1*(door_pos - 1.57)*(door_pos - 1.57)),
            # ('smooth', 0*-1e-5*np.sum(obs_dict['hand_vel']**2)), # hand_vel is not in obs. We can't compute this term for mbrl
            ('score', door_pos),
            ('bonus', 2*(door_pos > 0.2) + 8*(door_pos > 1.0) + 10*(door_pos > 1.35))
        ))
        # rwd_dict1 = rwd_dict.copy()
        # np.set_printoptions(precision=6)
        # print(self.squeeze_dims(rwd_dict1))

        self.rwd_dict["total"] = np.sum([self.rwd_dict[key] for key in RWD_KEYS], axis=0)
        self.rwd_dict["score"] = door_pos
        self.rwd_dict["solved"] = door_pos > 1.35
        self.rwd_dict["done"] = reach_dist > 50.0

        return self.rwd_dict
    
    # use latest obs, rwds to get all info (be careful, information belongs to different timestamps)
    # Its getting called twice. Once in step and sampler calls it as well
    def get_env_infos(self):
        env_info = {
            'time': self.obs_dict['t'][()],
            'reward': self.rwd_dict['total'][()],
            'score': self.rwd_dict['score'][()],
            'solved': self.rwd_dict['solved'][()],
            'done': self.rwd_dict['done'][()],
            'obs_dict': self.obs_dict,
            'rwd_dict': self.rwd_dict,
        }
        return env_info

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        act = paths["actions"]
        rewards = self.get_reward(obs, act)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid, 1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    # def get_env_infos(self):
    #     state = self.get_env_state()
    #     door_pos = self.data.qpos[self.door_hinge_did]
    #     goal_achieved = True if door_pos >= 1.35 else False
    #     return dict(state=state, goal_achieved=goal_achieved)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths, logger=None):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['solved']) > 5:
                num_success += 1
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            score = np.mean([np.mean(p['env_infos']['score']) for p in paths]) # return score/step
            logger.log_kv('score', score)
            logger.log_kv('success_rate', success_percentage)

        return success_percentage
