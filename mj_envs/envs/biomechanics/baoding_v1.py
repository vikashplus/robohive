import collections
import enum
import gym
import mujoco_py
import numpy as np

from mj_envs.envs.biomechanics.base_v0 import BaseV0
from mj_envs.envs.env_base import get_sim
from mj_envs.utils.quatmath import euler2quat
from mj_envs.utils.vectormath import calculate_cosine



## Define the task enum
class Task(enum.Enum):
    MOVE_TO_LOCATION = 0
    BAODING_CW = 1
    BAODING_CCW = 2

## Define task
# WHICH_TASK = Task.MOVE_TO_LOCATION
WHICH_TASK = Task.BAODING_CCW

class BaodingFixedEnvV1(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_pos', 'object1_pos', 'object1_velp', 'object2_pos', 'object2_velp', 'target1_pos', 'target2_pos']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
       'pos_dist_1':-5.0,
       'pos_dist_2':-5.0,
       'drop_penalty':-500.0,
       'wrist_angle':-0.5,
       'act_reg':-.1,
       'bonus':1.0
    }

    def __init__(self, model_path:str, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__ 
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we 
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path)

        self._setup(**kwargs)


    def _setup(self, 
               reward_option, 
               obs_keys=DEFAULT_OBS_KEYS, 
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, 
               **kwargs):

        # user parameters
        self.reward_option = reward_option
        self.which_task = Task(WHICH_TASK)
        self.drop_height_threshold = 0.85
        # self.wrist_threshold = 0.15
        self.proximity_threshold = 0.015
        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left
        self.ball_1_starting_angle = 3.*np.pi/4.0
        self.ball_2_starting_angle = -1*np.pi/4.0
        # init desired trajectory, for baoding
        self.x_radius = 0.025 #0.03
        self.y_radius = 0.028 #0.02 * 1.5 * 1.2
        self.center_pos = [-.0125, -.05] # [-.0020, -.0522]
        self.counter=0
        self.goal = self.create_goal_trajectory()

        # init target and body sites
        # self.grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.object1_sid = self.sim.model.site_name2id('ball1_site')
        self.object2_sid = self.sim.model.site_name2id('ball2_site')
        self.target1_sid = self.sim.model.site_name2id('target1_site')
        self.target2_sid = self.sim.model.site_name2id('target2_site')

        super()._setup(obs_keys=obs_keys, 
                       weighted_reward_keys=weighted_reward_keys, 
                       **kwargs)


    def step(self, a):
        desired_angle_wrt_palm = self.goal[self.counter].copy()
        desired_angle_wrt_palm[0] = desired_angle_wrt_palm[0] + self.ball_1_starting_angle
        desired_angle_wrt_palm[1] = desired_angle_wrt_palm[1] + self.ball_2_starting_angle

        desired_positions_wrt_palm = [0,0,0,0]
        desired_positions_wrt_palm[0] = self.x_radius*np.cos(desired_angle_wrt_palm[0]) + self.center_pos[0]
        desired_positions_wrt_palm[1] = self.y_radius*np.sin(desired_angle_wrt_palm[0]) + self.center_pos[1]
        desired_positions_wrt_palm[2] = self.x_radius*np.cos(desired_angle_wrt_palm[1]) + self.center_pos[0]
        desired_positions_wrt_palm[3] = self.y_radius*np.sin(desired_angle_wrt_palm[1]) + self.center_pos[1]

        # Updated desired
        self.sim.model.site_pos[self.target1_sid, 0] = desired_positions_wrt_palm[0]
        self.sim.model.site_pos[self.target1_sid, 1] = desired_positions_wrt_palm[1]
        self.sim.model.site_pos[self.target2_sid, 0] = desired_positions_wrt_palm[2]
        self.sim.model.site_pos[self.target2_sid, 1] = desired_positions_wrt_palm[3]
        # Updated desired (important to get consistent obs for both sims)
        self.sim_obsd.model.site_pos[self.target1_sid, 0] = desired_positions_wrt_palm[0]
        self.sim_obsd.model.site_pos[self.target1_sid, 1] = desired_positions_wrt_palm[1]
        self.sim_obsd.model.site_pos[self.target2_sid, 0] = desired_positions_wrt_palm[2]
        self.sim_obsd.model.site_pos[self.target2_sid, 1] = desired_positions_wrt_palm[3]

        self.counter +=1
        return super().step(a)

    def get_obs_vec(self):

        self.obs_dict['t']              = np.array([self.sim.data.time])
        self.obs_dict['hand_pos']       = self.sim.data.qpos[:-14].copy()

        # object positions
        # self.obs_dict['object1_pos']    = self.sim.data.qpos[-14:-11].copy()
        # self.obs_dict['object2_pos']    = self.sim.data.qpos[-7:-4].copy()
        self.obs_dict['object1_pos'] = self.sim.data.site_xpos[self.object1_sid].copy()
        self.obs_dict['object2_pos'] = self.sim.data.site_xpos[self.object2_sid].copy()

        # object translational velocities
        self.obs_dict['object1_velp']   = self.sim.data.qvel[-12:-9].copy()
        self.obs_dict['object2_velp']   = self.sim.data.qvel[-6:-3].copy()

        # site locations in world frame, populated after the step/forward call
        # self.obs_dict['target1_pos'] = np.array([self.sim.data.site_xpos[self.target1_sid][0], self.sim.data.site_xpos[self.target1_sid][1]])
        # self.obs_dict['target2_pos'] = np.array([self.sim.data.site_xpos[self.target2_sid][0], self.sim.data.site_xpos[self.target2_sid][1]])
        self.obs_dict['target1_pos'] = self.sim.data.site_xpos[self.target1_sid].copy()
        self.obs_dict['target2_pos'] = self.sim.data.site_xpos[self.target2_sid].copy()

        # object position error
        self.obs_dict['target1_err'] = self.obs_dict['target1_pos'] - self.obs_dict['object1_pos']
        self.obs_dict['target2_err'] = self.obs_dict['target2_pos'] - self.obs_dict['object2_pos']
        # self.obs_dict['target1_err'] = self.sim.data.site_xpos[self.target1_sid] - self.sim.data.site_xpos[self.object1_sid]
        # self.obs_dict['target2_err'] = self.sim.data.site_xpos[self.target2_sid] - self.sim.data.site_xpos[self.object2_sid]

        # muscle activations
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t']              = np.array([sim.data.time])
        obs_dict['hand_pos']       = sim.data.qpos[:-14].copy()

        # object positions
        # obs_dict['object1_pos']    = sim.data.qpos[-14:-11].copy()
        # obs_dict['object2_pos']    = sim.data.qpos[-7:-4].copy()
        obs_dict['object1_pos'] = sim.data.site_xpos[self.object1_sid].copy()
        obs_dict['object2_pos'] = sim.data.site_xpos[self.object2_sid].copy()

        # object translational velocities
        obs_dict['object1_velp']   = sim.data.qvel[-12:-9].copy()
        obs_dict['object2_velp']   = sim.data.qvel[-6:-3].copy()

        # site locations in world frame, populated after the step/forward call
        # obs_dict['target1_pos'] = np.array([sim.data.site_xpos[self.target1_sid][0], sim.data.site_xpos[self.target1_sid][1]])
        # obs_dict['target2_pos'] = np.array([sim.data.site_xpos[self.target2_sid][0], sim.data.site_xpos[self.target2_sid][1]])
        obs_dict['target1_pos'] = sim.data.site_xpos[self.target1_sid].copy()
        obs_dict['target2_pos'] = sim.data.site_xpos[self.target2_sid].copy()

        # object position error
        obs_dict['target1_err'] = obs_dict['target1_pos'] - obs_dict['object1_pos']
        obs_dict['target2_err'] = obs_dict['target2_pos'] - obs_dict['object2_pos']
        # obs_dict['target1_err'] = sim.data.site_xpos[self.target1_sid] - sim.data.site_xpos[self.object1_sid]
        # obs_dict['target2_err'] = sim.data.site_xpos[self.target2_sid] - sim.data.site_xpos[self.object2_sid]

        # muscle activations
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict['target1_err'], axis=-1)
        target2_dist = np.linalg.norm(obs_dict['target2_err'], axis=-1)

        # wrist pose err
        hand_pos = obs_dict['hand_pos'][:,:,:3] if obs_dict['hand_pos'].ndim==3 else obs_dict['hand_pos'][:3]
        wrist_pose_err = np.linalg.norm(hand_pos*np.array([5,0.5,1]), axis=-1)

        # detect fall
        object1_pos = obs_dict['object1_pos'][:,:,2] if obs_dict['object1_pos'].ndim==3 else obs_dict['object1_pos'][2]
        object1_pos = obs_dict['object2_pos'][:,:,2] if obs_dict['object2_pos'].ndim==3 else obs_dict['object2_pos'][2]
        is_fall_1 = object1_pos < self.drop_height_threshold
        is_fall_2 = object1_pos < self.drop_height_threshold
        is_fall = np.logical_or(is_fall_1, is_fall_2) #keep both balls up

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pos_dist_1',      target1_dist),
            ('pos_dist_2',      target2_dist),
            ('drop_penalty',    is_fall),
            ('wrist_angle',     wrist_pose_err),
            ('act_reg',         np.linalg.norm(self.obs_dict['act'], axis=-1)),
            ('bonus',           (target1_dist < self.proximity_threshold)+(target2_dist < self.proximity_threshold)+4*(target1_dist < self.proximity_threshold)*(target2_dist < self.proximity_threshold)),
            # Must keys
            ('sparse',          -target1_dist-target2_dist),
            ('solved',          (target1_dist < self.proximity_threshold)*(target2_dist < self.proximity_threshold)*(~is_fall)),
            ('done',            is_fall),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def reset_model(self, reset_pose=None, reset_vel=None, reset_goal=None):

        # reset counters
        self.counter=0

        # reset goal
        self.goal = self.create_goal_trajectory() if reset_goal is None else reset_goal.copy()

        # reset scene
        obs = super().reset_model(qp=reset_pose, qv=reset_vel)
        return obs

    def create_goal_trajectory(self):

        len_of_goals = 1000
        period = 200
        goal_traj = []
        if self.which_task==Task.BAODING_CW:
            sign = -1
        if self.which_task==Task.BAODING_CCW:
            sign = 1

        ### Reward option: continuous circle
        if self.reward_option==0:
            t = 0
            while t < len_of_goals:
                angle_before_shift = sign * 2 * np.pi * (t / period)
                goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                t += 1

        #### Reward option: increment in fourths
        elif self.reward_option==1:
            angle_before_shift = 0
            t = 0
            while t < len_of_goals:
                if(t>0 and t%50==0):
                    angle_before_shift += np.pi/2.0
                goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                t += 1

        #### Reward option: increment in eights
        elif self.reward_option==2:
            angle_before_shift = 0
            t = 0
            while t < len_of_goals:
                if(t>0 and t%30==0):
                    angle_before_shift += np.pi/4.0
                goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                t += 1

        goal_traj = np.array(goal_traj)

        return goal_traj

    # state utilities ========================================================
    def set_state(self, qpos=None, qvel=None, act=None, udd_state=None):
        """
        Set MuJoCo sim state
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        if qpos is None:
            qpos = old_state.qpos
        if qvel is None:
            qvel = old_state.qvel
        if act is None:
            act = old_state.act
        if udd_state is None:
            udd_state = old_state.udd_state
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, act, udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def get_env_state(self):
        """
        Get full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        act = self.sim.data.act.ravel().copy() if self.sim.model.na>0 else None
        mocap_pos = self.sim.data.mocap_pos.copy() if self.sim.model.nmocap>0 else None
        mocap_quat = self.sim.data.mocap_quat.copy() if self.sim.model.nmocap>0 else None
        site_pos = self.sim.model.site_pos[:].copy() if self.sim.model.nsite>0 else None
        body_pos = self.sim.model.body_pos[:].copy()
        return dict(qpos=qp, qvel=qv, act=act, mocap_pos=mocap_pos, mocap_quat=mocap_quat, site_pos=site_pos, body_pos=body_pos)


    def set_env_state(self, state_dict):
        """
        Set full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        act = state_dict['act']
        self.set_state(qp, qv, act)
        self.sim.model.site_pos[:] = state_dict['site_pos']
        self.sim.model.body_pos[:] = state_dict['body_pos']
        self.sim.forward()

        # evaluate paths and log metrics to logger
    def evaluate_success(self, paths, logger=None):
        num_success = 0
        num_paths = len(paths)
        horizon = self.spec.max_episode_steps # paths could have early termination

        # percentage of time balls were close to success
        for path in paths:
            num_success += np.sum(path['env_infos']['solved'][-5:], dtype=np.int)/self.spec.max_episode_steps
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
            rwd_dense = np.mean([np.sum(p['env_infos']['rwd_dense'])/horizon for p in paths]) # return rwd/step
            logger.log_kv('rwd_sparse', rwd_sparse)
            logger.log_kv('rwd_dense', rwd_dense)
            logger.log_kv('success_rate', success_percentage)

        return success_percentage


class BaodingRandomEnvV1(BaodingFixedEnvV1):

    def reset_model(self):
        # randomize target
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        # self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        obs = super().reset_model()
        return obs