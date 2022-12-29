""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
from mj_envs.utils.paths_utils import plot as plotnsave_paths
from mj_envs.utils.policies.rand_eef_policy import RandEEFPolicy
from mj_envs.utils.policies.heuristic_policy import HeuristicPolicy, HeuristicPolicyReal
from mj_envs.utils import tensor_utils
from PIL import Image
from pathlib import Path
import click
import numpy as np
import pickle
import time
import os
import glob
import threading
import copy
import cv2
import random
import torch

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots
USAGE:\n
    
    # Collect data
    $ python mj_envs/utils/collect_modem_rollouts_real.py -e FrankaPickPlaceRandom_v2d-v0 -n 10 -r none  -o /mnt/nfs_data/plancaster/remodem/demonstrations -on robopen07 -s 1 
    
    # Convert to h5
    $ python mj_envs/utils/paths_utils.py -u pickle2h5 -p /mnt/nfs_data/plancaster/remodem/demonstrations -e FrankaPickPlaceRandom_v2d-v0 -on robopen07_heuristic_rollouts -od /mnt/nfs_data/plancaster/remodem/datasets -vo True -hf dataset -cp True
    
    # Convert pickle to video
    $ python mj_envs/utils/paths_utils.py -u render -p /mnt/nfs_data/plancaster/remodem/demonstrations/robopen07_rollout0000000001.pickle -e FrankaPickPlaceRandom_v2d-v0 -rf mp4 
    
    # Playback actions from pickle
    $ python mj_envs/utils/examine_rollout.py -e FrankaPickPlaceRandom_v2d-v0 -p /mnt/nfs_data/plancaster/remodem/demonstrations/robopen07_rollout0000000001.pickle -m playback -r none -ea "{'is_hardware':True,'real':True}" 
'''

PIX_FROM_LEFT = 73
PIX_FROM_TOP = 58
X_DIST_FROM_CENTER = -1.0668/2
Y_DIST_FROM_BASE = 0.7493
Y_SCALE = -0.5207/152 # 0.5207 is length of bin 
X_SCALE = 1.0668/314 # 1.0668 is width of bin
OBJ_POS_LOW = [-0.25,0.368,0.91] #[-0.35,0.25,0.91]
OBJ_POS_HIGH = [0.25, 0.72, 0.91] #[0.35,0.65,0.91]
DROP_ZONE_LOW = [-0.18, 0.45, 1.15]
DROP_ZONE_HIGH = [0.18, 0.58, 1.15]
MOVE_JOINT_VEL = [0.15, 0.35, 0.25, 0.35, 0.3, 0.3, 0.45, 1.0, 1.0]
DIFF_THRESH = 0.45

MASK_START_X = 148 #40
MASK_END_X = 313 #400
MASK_START_Y = 57 #30
MASK_END_Y = 170 #220

def is_moving(prev, cur, tol):
    return np.linalg.norm(cur-prev) > tol

def rollout_policy(policy,
                   env,
                   start_o,
                   horizon=1000):
    """
        Examine a policy for behaviors;
        - either onscreen, or offscreen, or just rollout without rendering.
        - return resulting paths
    """

    # start rollouts

    observations=[]
    actions=[]
    rewards=[]
    env_infos = []

    o = start_o
    solved = False
    done = False
    t = 0
    ep_rwd = 0.0
    while t < horizon and not solved and not done:
        a = policy.get_action(o)[0]
        next_o, rwd, done, env_info = env.step(a)      
        #a = np.concatenate([env.last_ctrl, a])
        obs_dict = env.obsvec2obsdict(np.expand_dims(next_o, axis=(0,1)))
        solved = env_info['solved'] and obs_dict['qp'][0,0,7] > 0.001
        ep_rwd += rwd

        observations.append(o)
        actions.append(a)
        rewards.append(rwd)
        env_infos.append(env_info)
        o = next_o
        t = t+1



    path = dict(
    observations=np.array(observations),
    actions=np.array(actions),
    rewards=np.array(rewards),
    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    terminated=done
    )


    return observations[-1], path

def cart_move(action, env):
    last_pos = None
    action = 2*(((action-env.pos_limit_low)/(env.pos_limit_high-env.pos_limit_low))-0.5)
    old_vel_limit = env.vel_limit.copy()

    env.set_vel_limit(np.array(MOVE_JOINT_VEL))

    for _ in range(1000):

        obs, _, _, env_info = env.step(action)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos

    env.set_vel_limit(old_vel_limit)

# Adapted from here: https://github.com/fairinternal/robopen/blob/bb856233b3d8df89698ef80bf3c63ceab445b4aa/closed_loop_perception/robodev/policy/predict_grasp_release.py
def get_ccomp_grasp(img, out_dir, out_name):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    bin_mask = np.zeros(img.shape, dtype=np.uint8)

    bin_mask[MASK_START_Y:MASK_END_Y, MASK_START_X:MASK_END_X, :] = 255
    img_masked = cv2.bitwise_and(img, bin_mask)
    #img_masked_fn = os.path.join(out_dir, out_name+'_masked.png')
    #cv2.imwrite(img_masked_fn, img_masked)

    gray_img = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 
                                       255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       15,
                                       15)
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)

    # first box is background
    boxes = boxes[1:]
    filtered_boxes = []
    for x,y,w,h,pixels in boxes:
        if pixels < 1000 and h < 40 and w < 40 and h > 4 and w > 4:
            filtered_boxes.append((x,y,w,h))

    for x,y,w,h in filtered_boxes:
        cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,0,255), 1)

    rec_img_fn = os.path.join(out_dir, out_name+'recs.png')
    cv2.imwrite(rec_img_fn, img_masked)

    grasp_centers = []
    for x,y,w,h in filtered_boxes:
        grasp_x = X_SCALE*((x+(w/2.0)) - PIX_FROM_LEFT)+X_DIST_FROM_CENTER
        grasp_y = Y_SCALE*((y+(h/2.0)) - PIX_FROM_TOP) + Y_DIST_FROM_BASE
        if (grasp_x >= OBJ_POS_LOW[0] and grasp_x <= OBJ_POS_HIGH[0] and
            grasp_y >= OBJ_POS_LOW[1] and grasp_y <= OBJ_POS_HIGH[1]):
            grasp_centers.append((grasp_x,grasp_y))
    return grasp_centers

def dump_path(path, pfn):
    with open(pfn, 'wb') as pf:
        pickle.dump(path, pf)

def move_joint_config(env, config):
    last_pos = None

    config = 2*(((config-env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)
    old_vel_limit = env.vel_limit.copy()

    env.set_vel_limit(np.array(MOVE_JOINT_VEL))

    for _ in range(1000):

        obs, _, _, env_info = env.step(config)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos
    env.set_vel_limit(old_vel_limit)

    return obs, env_info

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
def main(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name, **{'reward_mode': 'sparse', 'is_hardware':True})
    env.seed(seed)

    pi = HeuristicPolicyReal(env, seed)

    # resolve directory
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)

    if (os.path.isdir(output_dir +'/frames') == False):
        os.mkdir(output_dir+'/frames')

    if output_name is None:
        output_name = 'heuristic'

    fps = glob.glob(str(Path(output_dir) / f'{output_name}*'))
    if len(fps) > 0:
        input('Output name {} found in {}, continue?'.format(output_name,output_dir))

    rollouts = 0
    successes = 0
    top_cam_key = None
    pickle_thread = None

    center_drop = np.array([0.0, -0.1, 0.0, -2.0, 0.0, 0.0, 0.0])
    left_drop = np.array([-0.01, 0.27, 0.59, -1.82, -0.16, 0.4, 0.66])
    right_drop = np.array([-0.51, 0.09, 0.03, -1.96, 0.0, 0.4, -0.47])
    drops_zones = [left_drop, center_drop, right_drop]

    #out_of_way = np.array([0.0, -1.04, 0.0, -2.3, 0.0, 1.0, -0.5])
    #out_of_way = np.array([0.5822, 0.2697, 0.01, -1.6753, -0.0067, 0.3391, 1.8494] )
    out_of_way = np.array([0.3438, -0.9361,  0.0876, -2.8211,  0.0749,  0.5144,  1.8283])
    max_gripper_open = 0.0002
    min_gripper_closed = 0.8228
    latest_img  = None

    filtered_boxes = None
    random_grasp_prob = 0.25

    while True:#successes < num_rollouts:

        obs = env.reset()

        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        open_qp = obs_dict['qp'][0,0,:9].copy()
        open_qp[7:9] = 0.0
        done = False

        while(obs_dict['qp'][0,0,7] > max_gripper_open and not done):
            release_action = 2*(((open_qp - env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)
            obs, _, done, env_info = env.step(release_action)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        # Detect and set the object pose
        if latest_img is not None:
            filtered_boxes = get_ccomp_grasp(latest_img, output_dir+'/debug', 'ccomp_grasp'+f'{(rollouts+seed):010d}')
            random.shuffle(filtered_boxes)
            latest_img = None

        if np.random.rand() > random_grasp_prob and filtered_boxes is not None and len(filtered_boxes) > 0:
            real_obj_pos = np.array([filtered_boxes[-1][0],filtered_boxes[-1][1], 0.91])
            filtered_boxes.pop()
            print('Vision obj pos')
        else:
            real_obj_pos = np.random.uniform(low=OBJ_POS_LOW, high=OBJ_POS_HIGH)
            print('Random obj pos')

        target_pos = np.array([-0.38, 0.5, 1.2])
        env.set_real_obj_pos(real_obj_pos)
        env.set_target_pos(target_pos)
        print('Real obs pos {} target pos {}'.format(real_obj_pos, target_pos))        

        #align_action = np.concatenate([real_obj_pos, [ 3.14,0.0,0.0,0.0,0.0]])
        #align_action[2] = 1.075
        #cart_move( align_action, env)

        print('Rolling out policy')
        env.set_slow_vel_limit(np.array([0.15, 0.15, 0.2, 0.15, 0.2, 0.2, 0.45, 1.0, 1.0]))
        obs, path = rollout_policy(pi,
                                   env,
                                   obs,
                                   horizon=100)#env.spec.max_episode_steps,)
        env.set_slow_vel_limit(np.array([0.15, 0.3, 0.2, 0.25, 0.2, 0.2, 0.35, 1.0, 1.0],))

        rollouts += 1

        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        if obs_dict['qp'][0,0,7] < max_gripper_open:
            print('Policy didnt close gripper, resetting')
            continue

        print('moving up')
        done = False
        des_grasp_pos = obs_dict['grasp_pos'][0,0,:].copy()
        des_grasp_pos[2] = 1.2        
        while(obs_dict['grasp_pos'][0,0,2] < 1.1 and not done):
            move_up_action = np.concatenate([des_grasp_pos, [3.14,0.0,0.0,obs_dict['qp'][0,0,7],0.0]])
            move_up_action = 2*(((move_up_action - env.pos_limit_low) / (env.pos_limit_high - env.pos_limit_low)) - 0.5)
            obs, _, done, _ = env.step(move_up_action)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))  

        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))      

        print('Grip Width {}'.format(obs_dict['qp'][0,0,7]))
        grip_width = obs_dict['qp'][0,0,7]
        mean_diff = 0.0
        if grip_width > max_gripper_open and grip_width < min_gripper_closed:
            obs, env_info = move_joint_config(env, np.concatenate([out_of_way, [obs_dict['qp'][0,0,7]]*2])) 

            '''
            done = False  
            tmp = 0      
            while(tmp < 100):
                move_out_action = np.concatenate([[-0.1,0.2,1.1], [3.14,0.0,0.0,obs_dict['qp'][0,0,7],0.0]])
                move_out_action = 2*(((move_out_action - env.pos_limit_low) / (env.pos_limit_high - env.pos_limit_low)) - 0.5)
                obs, _, done, env_info = env.step(move_out_action)
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
                print('qp: {}, grasp_pos: {}'.format(obs_dict['qp'][0,0,:7], obs_dict['grasp_pos'][0,0,:]))
                tmp+= 1
            if top_cam_key is None:
                for key in obs_dict.keys():
                    if 'top' in key:
                        top_cam_key = key
                        break            
            latest_img = env_info['obs_dict'][top_cam_key].copy()
            filtered_boxes = get_ccomp_grasp(latest_img, output_dir+'/debug', 'ccomp_grasp'+f'{(rollouts+seed):010d}')
            
            exit()
            '''
            # Wait for stabilize
            time.sleep(3)
            obs, env_info = move_joint_config(env, np.concatenate([out_of_way, [obs_dict['qp'][0,0,7]]*2]))   

            # Get top cam image
            if top_cam_key is None:
                for key in env_info['obs_dict'].keys():
                    if 'top' in key:
                        top_cam_key = key
                        break

            pre_drop_img = env_info['obs_dict'][top_cam_key]
            bin_mask = np.zeros(pre_drop_img.shape, dtype=np.uint8)

            bin_mask[MASK_START_Y:MASK_END_Y, MASK_START_X:MASK_END_X, :] = 255
            pre_drop_img = cv2.bitwise_and(pre_drop_img, bin_mask).astype(float)


            print('Moving to drop zone')
            drop_zone_pos = np.random.uniform(low=DROP_ZONE_LOW, high=DROP_ZONE_HIGH)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
            last_pos = None
            while(not done):
                drop_zone_action = np.concatenate([drop_zone_pos, [3.14,0.0,0.0,obs_dict['qp'][0,0,7],0.0]])
                drop_zone_action = 2*(((drop_zone_action - env.pos_limit_low) / (env.pos_limit_high - env.pos_limit_low)) - 0.5)
                obs, _, done, _ = env.step(drop_zone_action)
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))    
                pos = obs_dict['qp'][0,0,:7]
                if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
                    break
                last_pos = pos                

            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

            if np.linalg.norm(obs_dict['grasp_pos'][0,0,:2] - drop_zone_pos[:2]) > 0.1:
                continue

            open_qp = obs_dict['qp'][0,0,:9].copy()
            open_qp[7:9] = 0.0
            done = False

            print('Releasing')
            extra_time = 25
            while((obs_dict['qp'][0,0,7] > 0.001 and not done) or extra_time > 0):
                release_action = 2*(((open_qp - env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)
                obs, _, done, env_info = env.step(release_action)
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
                extra_time -= 1

            print('moving out of way after drop')
            obs, env_info = move_joint_config(env, np.concatenate([out_of_way, [obs_dict['qp'][0,0,7]]*2])) 
            time.sleep(3)
            obs, env_info = move_joint_config(env, np.concatenate([out_of_way, [obs_dict['qp'][0,0,7]]*2]))

            latest_img = env_info['obs_dict'][top_cam_key].copy()
            post_drop_img = cv2.bitwise_and(latest_img, bin_mask).astype(float)

            mean_diff = np.mean(np.abs(post_drop_img-pre_drop_img))
            print('Mean img diff: {}'.format(mean_diff))
        # Move to drop zone
        #drop_id = np.random.randint(low=0, high=len(drops_zones))
        #env.set_init_qpos(np.concatenate([drops_zones[drop_id], [0.0]]))

        # Determine success
        if mean_diff > DIFF_THRESH:
            ro_fn = output_name+'_rollout'+f'{(successes+seed):010d}'

            data = {}

            data['states'] = {}
            data['states']['qp'] = path['env_infos']['obs_dict']['qp']
            data['states']['qv'] = path['env_infos']['obs_dict']['qv']
            data['states']['grasp_pos'] = path['env_infos']['obs_dict']['grasp_pos']
            data['states']['object_err'] = path['env_infos']['obs_dict']['object_err']
            data['states']['target_err'] = path['env_infos']['obs_dict']['target_err']

            data['actions'] = path['actions']
            data['infos'] = [{'success': reward} for reward in path['rewards']]
            
            print('Saving frames')
            data['frames'] = []
            for cam in ['left_cam', 'right_cam', 'top_cam', 'Franka_wrist_cam']:
                rgb_key = 'rgb:'+cam+':240x424:2d'
                if rgb_key in env.obs_keys:
                    rgb_imgs = path['env_infos']['obs_dict'][rgb_key]
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
                    depth_imgs = path['env_infos']['obs_dict'][depth_key]
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

            '''
            path['env_infos']['obs_dict']['qp_arm'] = path['env_infos']['state']['qpos'][:,:7]
            path['env_infos']['obs_dict']['qv_arm'] = path['env_infos']['state']['qvel'][:,:7]
            path['env_infos']['obs_dict']['qp_ee'] = path['env_infos']['obs_dict']['grasp_pos']
            if 'observations' in path.keys():
                del path['observations']
            #if 'state' in path['env_infos'].keys():
            #    del path['env_infos']['state']
            #print('obs_dict.keys {}'.format(path['env_infos']['obs_dict'].keys()))
            #print('qp ee {}'.format(path['env_infos']['obs_dict']['qp_ee'].shape))
            #print('path[env_infos][obs_dict][t] {}'.format(path['env_infos']['obs_dict']['t'].shape))
            #print('path[actions] {}'.format(path['actions'].shape))
            #print('rgb:left_cam:240x424:2d {}'.format(path['env_infos']['obs_dict']['rgb:left_cam:240x424:2d'].shape))

            #dataset = paths_utils.path2dataset(path)

            pfn = output_dir+'/'+ro_fn+'.pickle'

            # Wait to finish previous pickle
            if pickle_thread is not None:
                print('Waiting for prev pickle to finish')
                pickle_thread.join()
            print('Starting pickle thread')
            pickle_thread = threading.Thread(target=dump_path, args=([copy.deepcopy(path)],pfn,))
            pickle_thread.start()
            '''
            '''
            data = {}
            data['states'] = path['observations'][:,:67]
            data['actions'] = path['actions']
            data['infos'] = [{'success': reward} for reward in path['rewards']]
            
            data['frames'] = []
            imgs = path['observations'][:,67:(67+240*424*3)]
            imgs = imgs.reshape((data['states'].shape[0],-1,240,424,3))
            imgs = imgs.astype(np.uint8)
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):
                    img_fn = ro_fn +'_cam'+str(j)+'_step'+f'{i:05d}'
                    img = Image.fromarray(imgs[i,j,:,:,:3])
                    img.save(output_dir+'/frames/'+img_fn+'.png')
                    if imgs.shape[1] == 1 or j == 1: # First case if there's only one cam, second case corresponds to right_cam
                        # Record path in data
                        data['frames'].append(Path(img_fn+'.png'))
         
            torch.save(data, output_dir+'/'+ro_fn+'.pt')
            '''

            successes += 1

        print('Success {} ({}/{})'.format(successes/rollouts,successes,rollouts))


if __name__ == '__main__':
    main()