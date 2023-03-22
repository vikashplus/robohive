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
        assert(len(img.shape)==3 and img.shape[2]==3, 'Incorrect img dims: {}'.format(img.shape))
        cropped_img = img[self.top_crop:self.bottom_crop, self.left_crop:self.right_crop, :]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        luv_img = cv2.cvtColor(np.array(cropped_img).astype('float32')/255, cv2.COLOR_RGB2Luv)
        luv_vec = np.mean(luv_img[:,:,1:],axis=(0,1))
        luv_angle = np.abs(np.dot(luv_vec,self.target_uv) / (np.linalg.norm(luv_vec)*np.linalg.norm(self.target_uv)))

        if self.render:
            bin_mask = 128*np.ones(img.shape, dtype=np.uint8)
            bin_mask[self.top_crop:self.bottom_crop, self.left_crop:self.right_crop, :] = 255
            img_masked = cv2.bitwise_and(img, bin_mask)            
            cv2.imshow("Success Detection", img_masked)
            cv2.waitKey(1)

        return luv_angle < self.thresh_val, luv_angle

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-c', '--cam_name', type=str, help='camera to get images from', default='top_cam')
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

  
if __name__ == '__main__':
    test_color_threshold_real()
