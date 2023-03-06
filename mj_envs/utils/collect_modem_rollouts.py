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

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots
USAGE:\n
    $ python examine_env.py --env_name door-v0 \n
    $ python examine_env.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''


# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='none')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-n', '--num_rollouts', type=int, help='number of rollouts to save', default=100)
@click.option('-sp', '--sparse_reward', type=bool, default=True)
@click.option('-pp', '--policy_path', type=str, default=None )
def collect_rollouts_cli(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, sparse_reward, policy_path):
    collect_rollouts(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, sparse_reward, policy_path)

def collect_rollouts(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, sparse_reward, policy_path):

    # seed and load environments
    np.random.seed(seed)
    if sparse_reward:
        env = gym.make(env_name, **{'reward_mode': 'sparse'})
    else:
        env = gym.make(env_name)        
    env.seed(seed)
 
    if policy_path != 'None':
        assert(os.path.exists(policy_path))
        pi = pickle.load(open(policy_path, 'rb'))
        output_name = 'policy'
    else:
        pi = HeuristicPolicy(env, seed)
        output_name = 'heuristic'

    # resolve directory
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)

    if (os.path.isdir(output_dir +'/frames') == False):
        os.mkdir(output_dir+'/frames')

    rollouts = 0
    successes = 0

    # Find env cams
    cams = set()
    for key in env.obs_keys:
        splits = key.split(':')
        if len(splits) == 4 and 'cam' in splits[1]:
            cams.add(splits[1])
    cams = list(cams)

    while successes < num_rollouts:

        # examine policy's behavior to recover paths
        paths = env.examine_policy(
            policy=pi,
            horizon=env.spec.max_episode_steps,
            num_episodes=1,
            frame_size=(640,480),
            mode=mode,
            output_dir=output_dir+'/',
            filename=output_name+str(successes),
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

            if env.rgb_encoder is None:
                rgb_img = Image.fromarray(np.zeros((10,10,3),dtype=np.uint8))
                rgb_img_fn = ro_fn + '_rgb_left_cam_step'
                rgb_img.save(output_dir+'/frames/'+rgb_img_fn+f'{0:05d}.png')
                for i in range(data['states']['qp'].shape[0]):
                    data['frames'].append(Path(rgb_img_fn+f'{0:05d}.png'))                
            else:
                for cam in cams:
                    rgb_key = 'rgb:'+cam+':224x224:2d'
                    rgb_imgs = paths[0]['env_infos']['obs_dict'][rgb_key]
                    depth_key = 'd:'+cam+':224x224:2d'
                    depth_imgs = paths[0]['env_infos']['obs_dict'][depth_key]
                    for i in range(rgb_imgs.shape[0]):
                        if len(data['frames']) <= i:
                            data['frames'].append([])
                        
                        rgb_img = Image.fromarray(rgb_imgs[i])
                        rgb_img_fn = ro_fn + '_rgb_'+cam+'_step'
                        rgb_img.save(output_dir+'/frames/'+rgb_img_fn+f'{i:05d}.png')
                        data['frames'][i].append(Path(rgb_img_fn+f'{i:05d}.png'))

                        depth_img = 255*depth_imgs[i]       
                        depth_img = Image.fromarray(depth_img)
                        depth_img = depth_img.convert("L")
                        depth_img_fn = ro_fn + '_depth_'+cam+'_step'
                        depth_img.save(output_dir+'/frames/'+depth_img_fn+f'{i:05d}.png')
                        data['frames'][i].append(Path(depth_img_fn+f'{i:05d}.png'))

            '''
            for cam in ['left_cam', 'right_cam', 'top_cam', 'Franka_wrist_cam']:
                rgb_key = 'rgb:'+cam+':240x424:2d'
                if rgb_key in env.obs_keys:
                    rgb_imgs = paths[0]['env_infos']['obs_dict'][rgb_key]
                    for i in range(rgb_imgs.shape[0]):
                        rgb_img = Image.fromarray(rgb_imgs[i])
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
            '''

            torch.save(data, output_dir+'/'+ro_fn+'.pt')
            successes += 1

            print('Success {} ({}/{})'.format(successes/rollouts,successes,rollouts))

if __name__ == '__main__':
    collect_rollouts_cli()
