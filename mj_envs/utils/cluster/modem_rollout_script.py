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

    if (os.path.isdir(output_dir +'/targets') == False):
        os.mkdir(output_dir+'/targets')

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
            data['states'] = paths[0]['observations'][:,:41]
            data['actions'] = paths[0]['actions']
            data['infos'] = [{'success': reward} for reward in paths[0]['rewards']]
            
            '''
            data['frames'] = []
            imgs = paths[0]['observations'][:,41:]
            imgs = imgs.reshape((data['states'].shape[0],-1,240,424,4))
            imgs = imgs.astype(np.uint8)
            for i in range(imgs.shape[0]):
                img_paths = []
                for j in range(imgs.shape[1]):
                    img_fn = ro_fn +'_cam'+str(j)+'_step'+f'{i:05d}'
                    img = Image.fromarray(imgs[i,j])
                    img.save(output_dir+'/frames/'+img_fn+'.png')
                    img_paths.append(Path(img_fn+'.png'))
                data['frames'].append(img_paths)
            '''
            for key in env.obs_keys:
                if 'target' in key:
                    if 'rgb:' in key:
                        target_fn = ro_fn + '_target_cam_rgb'
                        target_imgs = paths[0]['env_infos']['obs_dict'][key]
                        target_img = Image.fromarray(target_imgs[0])
                        #print('rgb target img max {}, min {}'.format(np.max(target_img),np.min(target_img)))
                    elif 'd:' in key:
                        target_fn = ro_fn + '_target_cam_depth'
                        target_imgs = paths[0]['env_infos']['obs_dict'][key]
                        target_img = Image.fromarray(target_imgs[0])
                        target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)
                        #print('depth target_img max {}, min {}'.format(np.max(target_img), np.min(target_img)))
                        target_img = target_img.convert("L")                        
                    else:
                        continue 
                    target_img.save(output_dir+'/targets/'+target_fn+'.png')
                    data['target'] = Path(target_fn+'.png')

            data['frames'] = [[]]*paths[0]['observations'].shape[0]
            for cam in ['left', 'right', 'top', 'wrist']:
                for key in env.obs_keys:
                    if cam in key:
                        imgs = paths[0]['env_infos']['obs_dict'][key]
                        if ('rgb:' not in key) and ('d:' not in key):
                            continue
 
                        for i in range(imgs.shape[0]):
                            if 'rgb:' in key:
                                img = Image.fromarray(imgs[i])
                                img_fn = ro_fn + '_rgb_cam_'+cam+'_step'
                            elif 'd:' in key:
                                img = Image.fromarray(imgs[i])
                                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                                img = img.convert("L")
                                img_fn = ro_fn + '_depth_cam_'+cam+'_step'                            
                                

                            img.save(output_dir+'/frames/'+img_fn+f'{i:05d}.png')
                            data['frames'][i].append(Path(img_fn+f'{i:05d}.png'))
                        

            torch.save(data, output_dir+'/'+ro_fn+'.pt')
            successes += 1

            print('Success {} ({}/{})'.format(successes/rollouts,successes,rollouts))

