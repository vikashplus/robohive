""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import enum
from robohive.utils import gym
import numpy as np

from robohive.envs import env_base
from robohive.utils.quat_math import rotVecQuat

# Define the task enum
class Task(enum.Enum):
    MOVE_TO_LOCATION = 0
    BAODING_CW = 1
    BAODING_CCW = 2
# Choose task
WHICH_TASK = Task.BAODING_CCW

class BaodingFixedEnvV1(env_base.MujocoEnv):

    DEFAULT_CREDIT = """\
    DAPG: Demo Augmented Policy Gradient; Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations
        {Aravind Rajeshwaran*, Vikash Kumar*}, Abhiskek Gupta, John Schulman, Emanuel Todorov, and Sergey Levine
        RSS-2018 | https://sites.google.com/view/deeprl-dexterous-manipulation
    """

    DEFAULT_OBS_KEYS = [
        'hand_pos',
        'object1_pos', 'object1_velp',
        'object2_pos', 'object2_velp',
        'target1_pos', 'target2_pos', # can be removed since we are adding err
        'target1_err', 'target2_err', # New in V1 (not tested for any adverse effects)
        ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
       'pos_dist_1':5.0,
       'pos_dist_2':5.0,
       'drop_penalty':500.0,
       'wrist_angle':0.5,   # V0: had -10 if wrist>0.15, V1 has contineous rewards (not tested for any adverse effects)
       'act_reg':1.0,       # new in V1
       'bonus':1.0          # new in V1
    }
    MOVE_TO_LOCATION_RWD_KEYS_AND_WEIGHTS = {
       'pos_dist_1':5.0,
       'drop_penalty':500.0,
       'act_reg':1.0,       # new in V1
       'bonus':1.0          # new in V1
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
        super().__init__(env_credits=self.DEFAULT_CREDIT, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)

    def _setup(self,
            frame_skip:int=40,
            n_shifts_per_period:int=-1, # n_shifts/ rotation for target update (-1 contineous)
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):

        # relax motor gains
        for sim in [self.sim, self.sim_obsd]:
            sim.model.actuator_gainprm[:] *= 0.1
            sim.model.actuator_biasprm[:] *= 0.1

        # user parameters
        self.which_task = Task(WHICH_TASK)
        self.drop_height_threshold = 0.06
        self.proximity_threshold = 0.015
        self.n_shifts_per_period = n_shifts_per_period

        # balls start at these angles
        #   1= yellow = bottom right
        #   2= pink = top left
        self.ball_1_starting_angle = 3.*np.pi/4.0
        self.ball_2_starting_angle = -1.*np.pi/4.0

        # init desired trajectory, for rotations
        if self.which_task!=Task.MOVE_TO_LOCATION:
            self.x_radius = 0.03
            self.y_radius = 0.02 * 1.5 * 1.2
            self.center_pos = [0.005, 0.06]
        self.counter=0
        self.goal = self.create_goal_trajectory(time_step=frame_skip*self.sim.model.opt.timestep, time_period=6)

        # init target and body sites
        self.palm_bid = self.sim.model.body_name2id('palm')
        self.object1_sid = self.sim.model.site_name2id('ball1_site')
        self.object2_sid = self.sim.model.site_name2id('ball2_site')
        self.target1_sid = self.sim.model.site_name2id('target1_site')
        self.target2_sid = self.sim.model.site_name2id('target2_site')

        # configure envs
        if self.which_task==Task.MOVE_TO_LOCATION:
            # move baoding targets out of the way
            for sim in [self.sim, self.sim_obsd]:
                sim.model.site_pos[self.target1_sid, 0] = 0
                sim.model.site_pos[self.target1_sid, 2] = 0.05
                sim.model.site_pos[self.target1_sid, 1] = 0
                sim.model.site_pos[self.target2_sid, 0] = 0
                sim.model.site_pos[self.target2_sid, 2] = 0.05
                sim.model.site_pos[self.target2_sid, 1] = 0
                # make 2nd object invisible
                object2_gid = self.sim.model.geom_name2id('ball2')
                sim.model.geom_rgba[object2_gid,3] = 0
                sim.model.site_rgba[self.object2_sid,3] = 0

            # update target1 to be move target
            self.target1_sid = self.sim.model.site_name2id('move_target_site')
            # update rewards to be move rewards
            weighted_reward_keys = self.MOVE_TO_LOCATION_RWD_KEYS_AND_WEIGHTS

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    frame_skip=frame_skip,
                    **kwargs,
                )

        # reset position
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        # self.init_qpos[:-14] *= 0 # Use fully open as init pos

        # V0: Centered the action space around key_qpos[0]. Not sure if it matter.
        # self.act_mid = self.init_qpos[:self.n_jnt].copy()
        # self.upper_rng = 0.9*(self.sim.model.actuator_ctrlrange[:,1]-self.act_mid)
        # self.lower_rng = 0.9*(self.act_mid-self.sim.model.actuator_ctrlrange[:,0])

        # V0: Used strict pos and velocity bounds (This matter and needs to implemented ???)
        # pos_bounds=[[-40, 40]] * self.n_dofs, #dummy
        # vel_bounds=[[-3, 3]] * self.n_dofs,

    def step(self, a, **kwargs):
        if self.which_task==Task.MOVE_TO_LOCATION:
            desired_pos = self.goal[self.counter].copy()
            # update both simhive with desired targets
            for sim in [self.sim, self.sim_obsd]:
                # update target 1
                sim.model.site_pos[self.target1_sid, 0] = desired_pos[0]
                sim.model.site_pos[self.target1_sid, 1] = desired_pos[1]
                sim.model.site_pos[self.target1_sid, 2] = desired_pos[2]+0.02
        else :
            # get desired targets
            desired_angle_wrt_palm = self.goal[self.counter].copy()
            desired_angle_wrt_palm[0] = desired_angle_wrt_palm[0] + self.ball_1_starting_angle
            desired_angle_wrt_palm[1] = desired_angle_wrt_palm[1] + self.ball_2_starting_angle

            ball1_desired_pos_wrt_palm = np.array([
                self.x_radius*np.cos(desired_angle_wrt_palm[0]) + self.center_pos[0],
                -0.045,
                self.y_radius*np.sin(desired_angle_wrt_palm[0]) + self.center_pos[1]
            ])
            ball2_desired_pos_wrt_palm = np.array([
                self.x_radius*np.cos(desired_angle_wrt_palm[1]) + self.center_pos[0],
                -0.045,
                self.y_radius*np.sin(desired_angle_wrt_palm[1]) + self.center_pos[1]
            ])

            # update both simhive with desired targets
            palm_pos =  self.sim.data.body_xpos[self.palm_bid]
            palm_quat =  self.sim.data.body_xquat[self.palm_bid]

            for sim in [self.sim, self.sim_obsd]:
                sim.model.site_pos[self.target1_sid,:] = palm_pos + rotVecQuat(ball1_desired_pos_wrt_palm, palm_quat)
                sim.model.site_pos[self.target2_sid,:] = palm_pos + rotVecQuat(ball2_desired_pos_wrt_palm, palm_quat)

        self.counter +=1
        # V0: mean center and scaled differently
        # a[a>0] = self.act_mid[a>0] + a[a>0]*self.upper_rng[a>0]
        # a[a<=0] = self.act_mid[a<=0] + a[a<=0]*self.lower_rng[a<=0]
        return super().step(a, **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_pos'] = sim.data.qpos[:-14].copy() # 7*2 for ball's free joint, rest should be hand

        # object positions
        obs_dict['object1_pos'] = sim.data.site_xpos[self.object1_sid].copy()
        obs_dict['object2_pos'] = sim.data.site_xpos[self.object2_sid].copy()

        # object translational velocities (V0 didn't normalize with dt)
        obs_dict['object1_velp'] = sim.data.qvel[-12:-9].copy()*self.dt
        obs_dict['object2_velp'] = sim.data.qvel[-6:-3].copy()*self.dt # there was a bug in V0 here

        # site locations in world frame, populated after the step/forward call
        obs_dict['target1_pos'] = sim.data.site_xpos[self.target1_sid].copy() # V0 was <x,y>, no z dim
        obs_dict['target2_pos'] = sim.data.site_xpos[self.target2_sid].copy() # V0 was <x,y>, no z dim

        # object position error
        obs_dict['target1_err'] = obs_dict['target1_pos'] - obs_dict['object1_pos'] # this wasn't a part of V0
        obs_dict['target2_err'] = obs_dict['target2_pos'] - obs_dict['object2_pos'] # this wasn't a part of V0

        # muscle activations
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict['target1_err'], axis=-1)
        target2_dist = np.linalg.norm(obs_dict['target2_err'], axis=-1)
        target_dist = target1_dist if self.which_task==Task.MOVE_TO_LOCATION else (target1_dist+target2_dist)
        if self.sim.model.na ==0:
            act_mag = np.array([[0]]) if obs_dict['hand_pos'].ndim==3 else 0
        else:
            act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na

        # wrist pose err (New in V1)
        hand_pos = obs_dict['hand_pos'][:,:,:3] if obs_dict['hand_pos'].ndim==3 else obs_dict['hand_pos'][:3]
        wrist_pose_err = np.linalg.norm(hand_pos*np.array([5,0.5,1]), axis=-1)
        # V0: penalize wrist angle for lifting up (positive) too much
        # wrist_threshold = 0.15
        # wrist_too_high = zeros
        # wrist_too_high[wrist_angle>wrist_threshold] = 1
        # self.reward_dict['wrist_angle'] = -10 * wrist_too_high

        # detect fall
        object1_pos = obs_dict['object1_pos'][:,:,2] if obs_dict['object1_pos'].ndim==3 else obs_dict['object1_pos'][2]
        object2_pos = obs_dict['object2_pos'][:,:,2] if obs_dict['object2_pos'].ndim==3 else obs_dict['object2_pos'][2]
        is_fall_1 = object1_pos < self.drop_height_threshold
        is_fall_2 = object2_pos < self.drop_height_threshold
        is_fall = is_fall_1 if self.which_task==Task.MOVE_TO_LOCATION else np.logical_or(is_fall_1, is_fall_2) # keep single/ both balls up

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pos_dist_1',      -1.*target1_dist), # V0 had only xy, V1 has xyz
            ('pos_dist_2',      -1.*target2_dist), # V0 had only xy, V1 has xyz
            ('drop_penalty',    -1.*is_fall),
            ('wrist_angle',     -1.*wrist_pose_err),    # V0 had -10 if wrist>0.15
            ('act_reg',         -1.*act_mag),
            ('bonus',           1.*(target1_dist < self.proximity_threshold)+1.*(target2_dist < self.proximity_threshold)+4.*(target1_dist < self.proximity_threshold)*(target2_dist < self.proximity_threshold)),
            # Must keys
            ('sparse',          -target_dist), # V0 had only xy, V1 has xyz
            ('solved',          (target1_dist < self.proximity_threshold)*(target2_dist < self.proximity_threshold)*(~is_fall)),
            ('done',            is_fall),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=6, **kwargs):
        # reset counters
        self.counter=0

        # reset goal
        self.goal = self.create_goal_trajectory(time_period=time_period) if reset_goal is None else reset_goal.copy()

        # reset scene
        obs = super().reset(reset_qpos=reset_pose, reset_qvel=reset_vel, **kwargs)
        return obs

    def create_goal_trajectory(self, time_step=.1, time_period=6):
        len_of_goals = 1000 # assumes that its greator than env horizon

        # populate go-to task with a target location
        if self.which_task==Task.MOVE_TO_LOCATION:
            goal_pos = self.np_random.choice([0,1,2,3])
            desired_position = []
            if goal_pos==0:
                desired_position.append(0.01)  #x
                desired_position.append(0.04)  #y
                desired_position.append(0.2)  #z
            elif goal_pos==1:
                desired_position.append(0)
                desired_position.append(-0.06)
                desired_position.append(0.24)
            elif goal_pos==2:
                desired_position.append(-0.02)
                desired_position.append(-0.02)
                desired_position.append(0.2)
            else:
                desired_position.append(0.03)
                desired_position.append(-0.02)
                desired_position.append(0.2)

            goal_traj = np.tile(desired_position, (len_of_goals, 1))

        # populate baoding task with a trajectory of goals to hit
        else:
            goal_traj = []
            if self.which_task==Task.BAODING_CW:
                sign = -1
            if self.which_task==Task.BAODING_CCW:
                sign = 1

            # Target updates in continuous circle
            if self.n_shifts_per_period==-1:
                t = 0
                while t < len_of_goals:
                    angle_before_shift = sign * 2 * np.pi * (t * time_step / time_period)
                    goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                    t += 1

            # Target updates in shifts (e.g. half turns, quater turns)
            else:
                t = 0
                angle_before_shift = 0
                steps_per_shift = np.ceil(time_period/(self.n_shifts_per_period*time_step)) # V0 had steps_per_fourth=7, which was twice faster than the period=60step
                while t < len_of_goals:
                    if(t>0 and t%steps_per_shift==0):
                        angle_before_shift += 2.0*np.pi/self.n_shifts_per_period
                    goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                    t += 1

            goal_traj = np.array(goal_traj)
        return goal_traj

class BaodingRandomEnvV1(BaodingFixedEnvV1):

    def reset(self, **kwargs):
        obs = super().reset(time_period = self.np_random.uniform(high=5, low=7), **kwargs)
        return obs
