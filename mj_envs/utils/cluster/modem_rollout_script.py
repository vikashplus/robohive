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
import click
import numpy as np
import pickle
import time
import os
import torch


# Random policy
class rand_policy():
    def __init__(self, env, seed):
        self.env = env
        self.env.action_space.np_random.seed(seed) # requires exlicit seeding

    def get_action(self, obs):
        # return self.env.np_random.uniform(high=self.env.action_space.high, low=self.env.action_space.low)
        return self.env.action_space.sample(), {'mode': 'random samples'}


# MAIN =========================================================
def collect_rollouts(job_data):
    env_name = job_data.env_name
    mode = job_data.mode
    seed = job_data.seed 
    render = job_data.render
    camera_name = job_data.camera_name
    output_dir = job_data.output_dir
    output_name = job_data.output_name
    num_rollouts = job_data.num_rollouts

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name, **{'reward_mode': 'sparse'})
    env.seed(seed)

    pi = HeuristicPolicy(env, seed)
    output_name = 'heuristic'

    # resolve directory
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)

    if (os.path.isdir(output_dir +'/frames') == False):
        os.mkdir(output_dir+'/frames')

    rollouts = 0
    successes = 0
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

            data['states'] = {}
            data['states']['qp'] = paths[0]['env_infos']['obs_dict']['qp']
            data['states']['qv'] = paths[0]['env_infos']['obs_dict']['qv']
            data['states']['grasp_pos'] = paths[0]['env_infos']['obs_dict']['grasp_pos']
            data['states']['object_err'] = paths[0]['env_infos']['obs_dict']['object_err']
            data['states']['target_err'] = paths[0]['env_infos']['obs_dict']['target_err']

            data['actions'] = paths[0]['actions']
            data['infos'] = [{'success': reward} for reward in paths[0]['rewards']]
            
            data['frames'] = []
            #for cam in ['left_cam', 'right_cam', 'top_cam', 'Franka_wrist_cam']:
            for cam in ['right_cam']:
                rgb_key = 'rgb:'+cam+':240x424:2d'
                if rgb_key in env.obs_keys:
                    rgb_imgs = paths[0]['env_infos']['obs_dict'][rgb_key]
                    for i in range(rgb_imgs.shape[0]):
                        print('WARN Cropping rgb image')
                        rgb_img = Image.fromarray(rgb_imgs[i,:224,:224,:])
                        rgb_img_fn = ro_fn + '_rgb_'+cam+'_step'
                        
                        rgb_img.save(output_dir+'/frames/'+rgb_img_fn+f'{i:05d}.png')
                        if len(data['frames']) <= i:
                            data['frames'].append([])
                        data['frames'][i].append(Path(rgb_img_fn+f'{i:05d}.png'))
                else:
                    print('WARN: Missing rgb key: {}'.format(rgb_key))
                    
                depth_key = 'd:'+cam+':240x424:2d'
                if depth_key in env.obs_keys:
                    depth_imgs = paths[0]['env_infos']['obs_dict'][depth_key]
                    for i in range(depth_imgs.shape[0]):
                        depth_img = 255*depth_imgs[i]       
                        depth_img = Image.fromarray(depth_img)
                        depth_img = depth_img.convert("L")
                        depth_img_fn = ro_fn + '_depth_'+cam+'_step'                            
                        depth_img.save(output_dir+'/frames/'+depth_img_fn+f'{i:05d}.png')
                        if len(data['frames']) <= i:
                            data['frames'].append([])
                        data['frames'][i].append(Path(depth_img_fn+f'{i:05d}.png'))
                else:
                    print('WARN: Missing depth key: {}'.format(depth_key))                        
         
            torch.save(data, output_dir+'/'+ro_fn+'.pt')
            successes += 1

            print('Success {} ({}/{})'.format(successes/rollouts,successes,rollouts))

