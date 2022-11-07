import gym
from mj_envs.utils.paths_utils import plot as plotnsave_paths
import numpy as np
import pickle

BEGIN_GRASP_THRESH = 0.08
BEGIN_DESCENT_THRESH = 0.05
ALIGN_HEIGHT = 1.25
GRIPPER_FULL_OPEN = 0.04
GRIPPER_FULL_CLOSE = 0.00
GRIPPER_BUFF_N = 4
GRIPPER_CLOSE_THRESH = 0.003

class HeuristicPolicy():
    def __init__(self, env, seed):
        self.env = env
        self.gripper_buff = GRIPPER_FULL_OPEN*np.ones((GRIPPER_BUFF_N,2))
        self.yaw = 0.0
        np.random.seed(seed)
        
    def get_action(self, obs):

        # TODO Change obsvec2dict to handle obs vectors with single dim
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        action = np.concatenate([obs_dict['grasp_pos'][0,0,:], [3.14,0.0,self.yaw], obs_dict['qp'][0,0,7:9]])
        
        if np.linalg.norm(obs_dict['object_err'][0,0,:]) > BEGIN_GRASP_THRESH:
            # Reset gripper buff
            self.gripper_buff = GRIPPER_FULL_OPEN*np.ones((GRIPPER_BUFF_N,2))

            # Object not yet within gripper
            if np.linalg.norm(obs_dict['object_err'][0,0,:2]) > BEGIN_DESCENT_THRESH:
                self.yaw = np.random.uniform(-3.14, 0.0)
                # Gripper not yet aligned with object (also open gripper)
                action[2] = ALIGN_HEIGHT
                action[:2] += obs_dict['object_err'][0,0,0:2]
                action[6:8] = GRIPPER_FULL_OPEN
        
            else:
                # Gripper aligned with object, go down towards it (also open gripper)
                action[:3] += obs_dict['object_err'][0,0,0:3]
                action[6:8] = GRIPPER_FULL_OPEN
        else:
            # Close gripper, move to target once gripper has had chance to close
            for i in range(self.gripper_buff.shape[0]-1):
                self.gripper_buff[i] = self.gripper_buff[i+1]
            self.gripper_buff[-1] = obs_dict['qp'][0,0,7:9]

            if np.all(np.linalg.norm(self.gripper_buff - GRIPPER_FULL_OPEN, axis=1) > 1e-4):
                # Move to target
                action[:3] += obs_dict['target_err'][0,0,0:3]

            action[6:8] = GRIPPER_FULL_CLOSE


            
        '''
        elif (np.linalg.norm(obs_dict['qp'][0,0,7:9] - GRIPPER_FULL_OPEN) < 1e-4 or
                np.linalg.norm(obs_dict['qv'][0,0,7:9]) > 1e-4):
            # Gripper either open or closing, tell it to close
            action[6:8] = GRIPPER_FULL_CLOSE
            print('Closing gripper')
        
        else:
            print('Move to target')
            # Object within gripper and gripper is closed, move to target pose
            #action[:3] += obs_dict['target_err'][0,0,0:3]
        '''
        
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
