import numpy as np
import pickle

class RandEEFPolicy():
    def __init__(self, seed):
        np.random.seed(seed)
        #low=[-0.435, 0.2, 1.0, 3.14, 0.0, -3.14, 0.0, 0.0]
        #high=[-0.035, 0.8, 1.0, 3.14, 0.0, 0.0, 0.04, 0.04]
        self.low=[0, 0.5, 0.85, 3.14, 0.0, 0.0, 0.0, 0.0]
        self.high=[0, 0.5, 1.25, 3.14, 0.0, 0.0, 0.04, 0.04]

    def get_action(self, obs):

        action = np.random.uniform(low=self.low,high=self.high)
        
        return action, {'evaluation': action}

if __name__ == '__main__':
    rep = RandEEFPolicy(seed=123)
    with open('./mj_envs/utils/policies/rep.pickle', 'wb') as fn:
        pickle.dump(rep, fn)

    pi = pickle.load(open('./mj_envs/utils/policies/rep.pickle', 'rb'))
    print('Loaded policy')
