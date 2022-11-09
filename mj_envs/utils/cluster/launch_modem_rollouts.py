""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
from mj_envs.utils.paths_utils import plot as plotnsave_paths
from mj_envs.utils.policies.rand_eef_policy import RandEEFPolicy
from mj_envs.utils.policies.heuristic_policy import HeuristicPolicy
from PIL import Image
from pathlib import Path
import numpy as np
import pickle
import time
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Random policy
class rand_policy():
    def __init__(self, env, seed):
        self.env = env
        self.env.action_space.np_random.seed(seed) # requires exlicit seeding

    def get_action(self, obs):
        # return self.env.np_random.uniform(high=self.env.action_space.high, low=self.env.action_space.low)
        return self.env.action_space.sample(), {'mode': 'random samples'}

@hydra.main(config_name='default', config_path='config')
def main(cfg:dict):
    print("Working directory : {}".format(os.getcwd()))
    env_name = cfg.env_name
    policy_path = cfg.policy_path
    mode = cfg.mode
    seed = cfg.seed 
    render = cfg.render
    camera_name = cfg.camera_name
    output_dir = cfg.output_dir
    output_name = cfg.output_name
    num_rollouts = cfg.num_rollouts
    
    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name, **{'reward_mode': 'sparse'})
    env.seed(seed)

    # resolve policy and outputs
    if policy_path is not None:
        pi = pickle.load(open(policy_path, 'rb'))
        if output_dir == './': # overide the default
            output_dir, pol_name = os.path.split(policy_path)
            output_name = os.path.splitext(pol_name)[0]
        if output_name is None:
            pol_name = os.path.split(policy_path)[1]
            output_name = os.path.splitext(pol_name)[0]
    else:
        pi = rand_policy(env, seed)
        mode = 'exploration'
        output_name ='random_policy'

    # resolve directory
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)

    if (os.path.isdir(output_dir +'/frames') == False):
        os.mkdir(output_dir+'/frames')

    rollouts = 0
    successes = 0
    act_low = np.array(env.pos_limit_low)
    act_high = np.array(env.pos_limit_high)
    while successes < num_rollouts:
        # examine policy's behavior to recover paths
        paths = env.examine_policy(
            policy=pi,
            horizon=env.spec.max_episode_steps,
            num_episodes=1,
            frame_size=(640,480),
            mode=mode,
            output_dir=output_dir+'/',
            filename=output_name,
            camera_name=camera_name,
            render=render)
        rollouts += 1

        # evaluate paths
        success_percentage = env.env.evaluate_success(paths)
        if success_percentage > 0.5:
            ro_fn = 'rollout'+f'{(successes+seed):010d}'

            data = {}
            data['states'] = paths[0]['observations'][:,:66]
            actions = 2*(((paths[0]['actions']-act_low)/(act_high-act_low))-0.5)
            data['actions'] = actions
            data['infos'] = [{'success': reward} for reward in paths[0]['rewards']]
            
            data['frames'] = []
            imgs = paths[0]['observations'][:,66:]
            imgs = imgs.reshape((data['states'].shape[0],-1,224,224,3))
            imgs = imgs.astype(np.uint8)
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):
                    img_fn = ro_fn +'_cam'+str(j)+'_step'+f'{i:05d}'
                    img = Image.fromarray(imgs[i,j])
                    img.save(output_dir+'/frames/'+img_fn+'.png')
                    if imgs.shape[1] == 1 or j == 1: # First case if there's only one cam, second case corresponds to right_cam
                        # Record path in data
                        data['frames'].append(Path(img_fn+'.png'))
         
            torch.save(data, output_dir+'/'+ro_fn+'.pt')
            successes += 1

            print('Success {} ({}/{})'.format(successes/rollouts,successes,rollouts))

if __name__ == '__main__':
    
    main()
