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
import cv2

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots
USAGE:\n
    $ python examine_env.py --env_name door-v0 \n
    $ python examine_env.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

class ColorThreshold():
    def __init__(self, cam_name, left_crop, right_crop, top_crop, bottom_crop, thresh_val, target_uv, render=None):
        self.cam_name = cam_name
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop
        self.thresh_val = thresh_val
        self.target_uv = target_uv
        self.render = render

    def detect_success(self, img):
        cropped_img = img[self.top_crop:self.bottom_crop, self.left_crop:self.right_crop, :]
        luv_img = cv2.cvtColor(np.array(cropped_img).astype('float32')/255, cv2.COLOR_RGB2Luv)
        luv_vec = np.mean(luv_img[:,:,1:],axis=(0,1))
        luv_angle = np.dot(luv_vec,self.target_uv) / (np.linalg.norm(luv_vec)*np.linalg.norm(self.target_uv))

        if self.render:
            bin_mask = 128*np.ones(img.shape, dtype=np.uint8)
            bin_mask[self.top_crop:self.bottom_crop, self.left_crop:self.right_crop, :] = 255
            img_masked = cv2.bitwise_and(img, bin_mask)            
            cv2.imshow("Success Detection", img_masked)
            cv2.waitKey(1)

        return luv_angle < self.thresh_val

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-c', '--cam_name', tpye=str, help='camera to get images from', default='top_cam')
def test_color_threshold_real(env_name, seed, cam_name):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name, **{'is_hardware':True})      
    env.seed(seed)

    ct = ColorThreshold(cam_name=cam_name,
                        left_crop=32,
                        right_crop=197,
                        top_crop=41,
                        bottom_crop=154,
                        thresh_val=1.0,
                        target_uv=np.array([0,0.55]))

    o = env.reset()
    eef_cmd = env.last_eef_cmd
    eef_cmd = (0.5 * eef_cmd.flatten() + 0.5) * (env.pos_limit_high - env.pos_limit_low) + env.pos_limit_low
    while True:  
        next_o, rwd, done, env_info = env.step(eef_cmd)
        rgb_key = 'rgb:'+cam_name+':224x224:2d'
        rgb_img = env_info['obs_dict'][rgb_key]
        print(ct.detect_success())

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
