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
    $ python mj_envs/utils/collect_modem_rollouts_real.py -e FrankaPickPlaceRandomReal_v2d-v0 -n 1 -r none  -o /mnt/nfs_code/robopen_users/plancaster/remodem/demonstrations/franka-FrankaPickPlaceRandomReal_v2d -on robopen07 -s 1
    
    # Convert to h5
    $ python mj_envs/utils/paths_utils.py -u pickle2h5 -p /mnt/nfs_data/plancaster/remodem/demonstrations -e FrankaPickPlaceRandom_v2d-v0 -on robopen07_heuristic_rollouts -od /mnt/nfs_data/plancaster/remodem/datasets -vo True -hf dataset -cp True
    
    # Convert pickle to video
    $ python mj_envs/utils/paths_utils.py -u render -p /mnt/nfs_data/plancaster/remodem/demonstrations/robopen07_rollout0000000001.pickle -e FrankaPickPlaceRandom_v2d-v0 -rf mp4 
    
    # Playback actions from pickle
    $ python mj_envs/utils/examine_rollout.py -e FrankaPickPlaceRandom_v2d-v0 -p /mnt/nfs_data/plancaster/remodem/demonstrations/robopen07_rollout0000000001.pickle -m playback -r none -ea "{'is_hardware':True,'real':True}" 
'''

PIX_FROM_LEFT = 73-116
PIX_FROM_TOP = 58-16
X_DIST_FROM_CENTER = -1.0668/2
Y_DIST_FROM_BASE = 0.72#0.7493
Y_SCALE = -0.5207/152 # 0.5207 is length of bin 
X_SCALE = 1.0668/314 # 1.0668 is width of bin
OBJ_POS_LOW = [-0.25,0.368,0.91] #[-0.35,0.25,0.91]
OBJ_POS_HIGH = [0.25, 0.72, 0.91] #[0.35,0.65,0.91]
DROP_ZONE = np.array([0.0, 0.53, 1.1])
#DROP_ZONE_PERTURB = np.array([0.18, 0.1,0.0])
DROP_ZONE_PERTURB = np.array([0.025, 0.025,0.0])
#DROP_ZONE_LOW = [-0.18, 0.43, 1.1]
#DROP_ZONE_HIGH = [0.18, 0.63, 1.1]
MOVE_JOINT_VEL = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.45, 1.0, 1.0]
DIFF_THRESH = 0.15#0.45

MASK_START_X = 148-116 #40
MASK_END_X = 313-116 #400
MASK_START_Y = 57-16 #30
MASK_END_Y = 170-16 #220

MAX_GRIPPER_OPEN = 0.0002
MIN_GRIPPER_CLOSED = 0.8228

OUT_OF_WAY = np.array([0.3438, -0.9361,  0.0876, -2.8211,  0.0749,  0.5144,  1.8283])
#OUT_OF_WAY = np.array([0.3438, -1.1,  0.0876, -2.5,  0.0749,  0.2,  1.8283])

def is_moving(prev, cur, tol):
    return np.linalg.norm(cur-prev) > tol

def rollout_policy(policy,
                   env,
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

    o = env.get_obs()

    env.squeeze_dims(env.rwd_dict)
    env.squeeze_dims(env.obs_dict)
    env_info = env.get_env_infos()

    observations.append(o)
    env_infos.append(env_info)

    solved = False
    done = False
    t = 0
    ep_rwd = 0.0
    while t < horizon:# and not done:
        a = policy.get_action(o)[0]
        next_o, rwd, done, env_info = env.step(a)      
        #a = np.concatenate([env.last_ctrl, a])
        obs_dict = env.obsvec2obsdict(np.expand_dims(next_o, axis=(0,1)))
        solved = env_info['solved'] and obs_dict['qp'][0,0,7] > 0.001
        ep_rwd += rwd

        observations.append(next_o)
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

    for _ in range(1000):

        obs, _, _, env_info = env.step(action)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos

def dump_path(path, pfn):
    with open(pfn, 'wb') as pf:
        pickle.dump(path, pf)

def move_joint_config(env, config):
    last_pos = None

    config = 2*(((config-env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)

    for _ in range(1000):

        obs, _, _, env_info = env.step(config)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos

    return obs, env_info

def open_gripper(env, obs):
    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    open_qp = obs_dict['qp'][0,0,:9].copy()
    open_qp[7:9] = 0.0

    start_time = time.time()

    while((obs_dict['qp'][0,0,7] > 0.001) and time.time()-start_time < 30.0):
        release_action = 2*(((open_qp - env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)
        obs, _, done, env_info = env.step(release_action)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    return obs

def check_grasp_success(env, obs, force_img=False, just_drop=False):
        failed_grasp = False
        if obs is None:
            obs = env.get_obs()
        
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        if obs_dict['qp'][0,0,7] < MAX_GRIPPER_OPEN:
            failed_grasp = True
            print('Policy didnt close gripper')
            if not force_img:
                return None, None, False, None, None

        if obs_dict['grasp_pos'][0,0,2] < 1.0:
            failed_grasp = True
            print('Policy didnt lift gripper')
            obs = open_gripper(env, obs)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
            if not force_img:
                return None, None, False, None, None

        print('moving up')
        des_grasp_pos = obs_dict['grasp_pos'][0,0,:].copy()
        des_grasp_y = des_grasp_pos[1]
        des_grasp_height = 1.2        
        move_up_tries = 4
        for i in range(1,move_up_tries+1):
            
            move_up_steps = 0
            while(obs_dict['grasp_pos'][0,0,2] < 1.1 and move_up_steps < 25):
                move_up_action = np.concatenate([des_grasp_pos, [3.14,0.0,0.0,obs_dict['qp'][0,0,7],0.0]])
                des_grasp_pos[1] = ((move_up_tries-i)/(move_up_tries))*des_grasp_y+((i)/(move_up_tries))*(DROP_ZONE[1]) 
                des_grasp_pos[1] = min(max(des_grasp_pos[1], obs_dict['grasp_pos'][0,0,1]-0.1),obs_dict['grasp_pos'][0,0,1]+0.1) 
                move_up_action[2] = min(1.2, obs_dict['grasp_pos'][0,0,2]+0.1)
                move_up_action = 2*(((move_up_action - env.pos_limit_low) / (env.pos_limit_high - env.pos_limit_low)) - 0.5)
                obs, _, done, _ = env.step(move_up_action)
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1))) 
                move_up_steps += 1 
            if obs_dict['grasp_pos'][0,0,2] >= 1.1:
                break

        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))      

        print('Grip Width {}'.format(obs_dict['qp'][0,0,7]))
        grip_width = obs_dict['qp'][0,0,7]
        mean_diff = 0.0

        if (grip_width < MAX_GRIPPER_OPEN or grip_width > MIN_GRIPPER_CLOSED):
            failed_grasp = True
            obs = open_gripper(env, obs)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
            if not force_img:
                return None, None, False, None, None

        pre_drop_img = None
        # Get top cam key
        top_cam_key = None
        for key in obs_dict.keys():
            if 'top' in key:
                top_cam_key = key
                break
        assert(top_cam_key is not None)

        if not just_drop and not failed_grasp:

            obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]*2])) 

            # Wait for stabilize
            time.sleep(3)
            obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]*2]))   

            pre_drop_img = env_info['obs_dict'][top_cam_key]

        if just_drop or not failed_grasp:

            print('Moving to drop zone')
            drop_zone_pos = np.random.uniform(low=DROP_ZONE-DROP_ZONE_PERTURB, high=DROP_ZONE+DROP_ZONE_PERTURB)
            drop_zone_yaw = -np.pi*np.random.rand()
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
            last_pos = None
            drop_zone_steps = 0
            while(drop_zone_steps < 100):
                drop_zone_action = np.concatenate([drop_zone_pos, [3.14,0.0,drop_zone_yaw,obs_dict['qp'][0,0,7],0.0]])
                drop_zone_action = 2*(((drop_zone_action - env.pos_limit_low) / (env.pos_limit_high - env.pos_limit_low)) - 0.5)
                obs, _, done, _ = env.step(drop_zone_action)
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))    
                pos = obs_dict['qp'][0,0,:7]
                if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
                    break
                last_pos = pos         
                drop_zone_steps += 1       

            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

            if np.linalg.norm(obs_dict['grasp_pos'][0,0,:2] - drop_zone_pos[:2]) > 0.1:
                #return None, None
                print('Rand drop failed, moving to init qpos for drop')
                obs, env_info = move_joint_config(env, np.concatenate([env.init_qpos[:7], [obs_dict['qp'][0,0,7]]*2]))
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

            open_qp = obs_dict['qp'][0,0,:9].copy()
            open_qp[7:9] = 0.0

            print('Releasing')
            extra_time = 25
            start_time = time.time()
            while(((obs_dict['qp'][0,0,7] > 0.001) or extra_time > 0) and time.time()-start_time < 30.0):
                release_action = 2*(((open_qp - env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)
                obs, _, done, env_info = env.step(release_action)
                obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
                extra_time -= 1

            drop_pos = obs_dict['grasp_pos'][0,0]
            drop_x = int(PIX_FROM_LEFT + (drop_pos[0]-X_DIST_FROM_CENTER)/X_SCALE)
            drop_y = int(PIX_FROM_TOP + (drop_pos[1] - Y_DIST_FROM_BASE)/Y_SCALE)
            print('drop_pos x: {}, drop_pos y: {}'.format(drop_zone_pos[0], drop_zone_pos[1]))
            if pre_drop_img is not None:
                success_mask = np.zeros(pre_drop_img.shape, dtype=np.uint8)
                success_start_x = max(MASK_START_X, drop_x - 30)
                success_end_x = min(MASK_END_X, drop_x + 30)
                success_start_y = max(MASK_START_Y, drop_y - 30)
                success_end_y = min(MASK_END_Y, drop_y+30)
                
                print('drop_x {}, drop_y {}, start_x {}, end_x {}, start_y {}, end_y {}'.format( drop_x, drop_y, success_start_x, success_end_x, success_start_y, success_end_y))
                success_mask[success_start_y:success_end_y, success_start_x:success_end_x, :] = 255
                pre_drop_img = cv2.bitwise_and(pre_drop_img, success_mask)

        latest_img = None
        post_drop_img = None
        if (not just_drop and not failed_grasp) or force_img:
            print('moving out of way')
            obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]*2])) 
            time.sleep(3)
            obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]*2]))

            latest_img = env_info['obs_dict'][top_cam_key].copy()

        if pre_drop_img is not None and latest_img is not None and success_mask is not None:
            post_drop_img = cv2.bitwise_and(latest_img, success_mask)
            mean_diff = np.mean(np.abs(post_drop_img.astype(float)-pre_drop_img.astype(float)))
            print('Mean img diff: {}'.format(mean_diff))    
        else:
            mean_diff =  0.0

        return mean_diff, latest_img, mean_diff > DIFF_THRESH, pre_drop_img, post_drop_img

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
@click.option('-fc','--force_check', type=bool, default=False)
def main(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, force_check):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name, **{'reward_mode': 'sparse', 'is_hardware':True})
    env.seed(seed)

    random_grasp_prob = 0.01

    pi = HeuristicPolicyReal(env, 
                             seed,
                             random_grasp_prob=random_grasp_prob, 
                             output_dir=output_dir)

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
    pickle_thread = None

    center_drop = np.array([0.0, -0.1, 0.0, -2.0, 0.0, 0.0, 0.0])
    left_drop = np.array([-0.01, 0.27, 0.59, -1.82, -0.16, 0.4, 0.66])
    right_drop = np.array([-0.51, 0.09, 0.03, -1.96, 0.0, 0.4, -0.47])
    drops_zones = [left_drop, center_drop, right_drop]

    latest_img  = None

    grasp_centers = None
    filtered_boxes = None
    img_masked = None
    print('Starting')
    while True:#successes < num_rollouts:

        if latest_img is not None and (pi.grasp_centers is None or len(pi.grasp_centers) <= 0):
            pi.update_grasps(img=latest_img,
                             out_dir=output_dir+'/debug')
        obs, path = pi.do_rollout(horizon=env.spec.max_episode_steps)
        rollouts += 1

        mean_diff, new_img, grasp_success, pre_drop_img, post_drop_img = check_grasp_success(env, obs, 
                                                                                            force_img=force_check or grasp_centers is None or len(grasp_centers) <= 0 )

        if mean_diff is None:
            continue
        else:
            latest_img = new_img

        # Determine success
        if grasp_success:

            ro_fn = output_name+'_rollout'+f'{(successes+seed):010d}'

            pre_img_fn = os.path.join(output_dir+'/debug',ro_fn+'pre.png')
            post_img_fn = os.path.join(output_dir+'/debug',ro_fn+'post.png')
            cv2.imwrite(pre_img_fn, cv2.cvtColor(pre_drop_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(post_img_fn, cv2.cvtColor(post_drop_img, cv2.COLOR_RGB2BGR))

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
                rgb_key = 'rgb:'+cam+':224x224:2d'
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
                    
                depth_key = 'd:'+cam+':224x224:2d'
                if depth_key in env.obs_keys:
                    depth_imgs = path['env_infos']['obs_dict'][depth_key]
                    for i in range(depth_imgs.shape[0]):
                        '''
                        import matplotlib.pyplot as plt
                        print('min {}, max {}'.format(np.min(depth_imgs[i]), np.max(depth_imgs[i])))
                        flat = depth_imgs[i].flatten()
                        print('mean {}, std {}'.format(np.mean(flat), np.std(flat)))
                        plt.hist(depth_imgs[i].flatten(),bins=25)
                        plt.savefig('/tmp/depth_hist.png')
                        exit()
                        '''

                        #depth_img = np.array(depth_imgs[i]/256, dtype=np.uint8)       
                        depth_img = np.array(np.clip(depth_imgs[i],0,255), dtype=np.uint8)       
                        depth_img = Image.fromarray(depth_img)
                        #depth_img = depth_img.convert("L")
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
            #print('rgb:left_cam:224x224:2d {}'.format(path['env_infos']['obs_dict']['rgb:left_cam:224x224:2d'].shape))

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
            imgs = path['observations'][:,67:(67+224*224*3)]
            imgs = imgs.reshape((data['states'].shape[0],-1,224,224,3))
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
