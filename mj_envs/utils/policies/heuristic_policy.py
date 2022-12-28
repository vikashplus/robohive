import gym
from mj_envs.utils.paths_utils import plot as plotnsave_paths
import numpy as np
import pickle

BEGIN_GRASP_THRESH = 0.08
SIM_BEGIN_DESCENT_THRESH = 0.05
REAL_BEGIN_DESCENT_THRESH = 0.08
SIM_ALIGN_HEIGHT = 1.185
REAL_ALIGN_HEIGHT = 1.075
SIM_GRIPPER_FULL_OPEN = 0.04
SIM_GRIPPER_FULL_CLOSE = 0.00
REAL_GRIPPER_FULL_OPEN = 0.00
REAL_GRIPPER_FULL_CLOSE = 0.04
GRIPPER_BUFF_N = 4
GRIPPER_CLOSE_THRESH = 1e-8
MOVE_THRESH = 0.005

class HeuristicPolicy():
    def __init__(self, env, seed):
        self.env = env
        self.gripper_buff = SIM_GRIPPER_FULL_OPEN*np.ones((GRIPPER_BUFF_N,2))
        self.yaw = 0.0
        np.random.seed(seed)
        
    def get_action(self, obs):

        # TODO Change obsvec2dict to handle obs vectors with single dim
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        action = np.concatenate([obs_dict['grasp_pos'][0,0,:], [3.14,0.0,self.yaw], obs_dict['qp'][0,0,7:9]])
        
        if np.linalg.norm(obs_dict['object_err'][0,0,:]) > BEGIN_GRASP_THRESH:
            # Reset gripper buff
            self.gripper_buff = SIM_GRIPPER_FULL_OPEN*np.ones((GRIPPER_BUFF_N,2))

            # Object not yet within gripper
            if np.linalg.norm(obs_dict['object_err'][0,0,:2]) > SIM_BEGIN_DESCENT_THRESH:

                self.yaw = np.random.uniform(-3.14, 0.0)
                # Gripper not yet aligned with object (also open gripper)
                action[2] = SIM_ALIGN_HEIGHT
                action[:2] += obs_dict['object_err'][0,0,0:2]
                action[6:8] = SIM_GRIPPER_FULL_OPEN
        
            else:
                # Gripper aligned with object, go down towards it (also open gripper)
                action[:3] += obs_dict['object_err'][0,0,0:3]
                action[6:8] = SIM_GRIPPER_FULL_OPEN
        else:
            # Close gripper, move to target once gripper has had chance to close
            for i in range(self.gripper_buff.shape[0]-1):
                self.gripper_buff[i] = self.gripper_buff[i+1]
            self.gripper_buff[-1] = obs_dict['qp'][0,0,7:9]

            if np.all(np.linalg.norm(self.gripper_buff - SIM_GRIPPER_FULL_OPEN, axis=1) > 1e-4):
                # Move to target
                action[:3] += obs_dict['target_err'][0,0,0:3]

            action[6:8] = SIM_GRIPPER_FULL_CLOSE


        # Normalize action to be between -1 and 1
        action = 2*(((action - self.env.pos_limit_low) / (self.env.pos_limit_high - self.env.pos_limit_low)) - 0.5)
        
        return action, {'evaluation': action}

class HeuristicPolicyReal():
    def __init__(self, env, seed):
        self.env = env

        self.yaw = 0.0
        self.stage = 0
        self.last_t = 0.0
        self.last_qp = None
        np.random.seed(seed)

    def get_action(self, obs):

        # TODO Change obsvec2dict to handle obs vectors with single dim
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        action = np.concatenate([obs_dict['grasp_pos'][0,0,:], [3.14,0.0,self.yaw], [obs_dict['qp'][0,0,7:9]]])

        # Figure out which stage we are in
        if self.last_t > obs_dict['t'][0,0,0]:
            # Reset
            self.stage = 0
            self.last_qp = None
            self.yaw = np.random.uniform(low = 0.0, high = 3.14)
        elif self.stage == 0: # Wait until aligned xy
            # Advance to next stage?
            if (np.linalg.norm(obs_dict['object_err'][0,0,:2]) < REAL_BEGIN_DESCENT_THRESH and
                (self.last_qp is not None and np.linalg.norm(obs_dict['qp'][0,0,:] - self.last_qp) < MOVE_THRESH)):
                self.stage = 1
        elif self.stage == 1:# Wait until close pregrasp
            if (np.linalg.norm(obs_dict['object_err'][0,0,:3]) < BEGIN_GRASP_THRESH or
                (self.last_qp is not None and np.linalg.norm(obs_dict['qp'][0,0,:] - self.last_qp) < MOVE_THRESH)):
                self.stage = 2
        elif self.stage == 2: # Wait until pregrasp has stabilized
            # Advance to next stage?
            if (self.last_qp is not None and np.linalg.norm(obs_dict['qp'][0,0,:] - self.last_qp) < MOVE_THRESH):
                self.stage = 3
        elif self.stage == 3: # Wait for gripper to start closing
            # Advance to next stage?
            if (self.last_qp is not None and obs_dict['qp'][0,0,7] > self.last_qp[7]):
                self.stage = 4
        elif self.stage == 4: # Wait for gripper to stop closing
            if (self.last_qp is not None and np.abs(self.last_qp[7] - obs_dict['qp'][0,0,7]) < GRIPPER_CLOSE_THRESH):
                self.stage = 5   

        self.last_t = obs_dict['t'][0,0,0]
        self.last_qp = obs_dict['qp'][0,0,:]

        #print('Stage {}, t {}'.format(self.stage, self.last_t))
        if self.stage == 0: # Align in xy
            action[2] = REAL_ALIGN_HEIGHT
            action[:2] += obs_dict['object_err'][0,0,0:2]
            action[6:8] = REAL_GRIPPER_FULL_OPEN
        elif self.stage == 1 or self.stage == 2: # Move to pregrasp
            action[:3] += obs_dict['object_err'][0,0,0:3]
            action[6:8] = REAL_GRIPPER_FULL_OPEN
        elif self.stage == 3 or self.stage == 4: # Close gripper
            action[:3] += obs_dict['object_err'][0,0,0:3]
            action[6:8] = REAL_GRIPPER_FULL_CLOSE
            #print('Grasp pos {}'.format(obs_dict['grasp_pos'][0,0,:]))
        elif self.stage == 5: # Move to target pose
            action[:3] += obs_dict['object_err'][0,0,0:3] #obs_dict['target_err'][0,0,0:3]
            action[6:8] = REAL_GRIPPER_FULL_CLOSE


        # Normalize action to be between -1 and 1
        action = 2*(((action - self.env.pos_limit_low) / (self.env.pos_limit_high - self.env.pos_limit_low)) - 0.5)

        return action, {'evaluation': action}

if __name__ == '__main__':
    env_name = 'FrankaPickPlaceRandom-v0'
    seed = 123
    np.random.seed(seed)
    env = gym.make(env_name) 
    env.seed(seed)

    hp = HeuristicPolicy(env=env, seed=seed)
    with open('./mj_envs/utils/policies/heuristic.pickle', 'wb') as fn:
        pickle.dump(hp, fn)

    pi = pickle.load(open('./mj_envs/utils/policies/heuristic.pickle', 'rb'))
    print('Loaded policy')
    obs =  env.reset()
    pi.get_action(obs)
