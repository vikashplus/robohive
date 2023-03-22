import gym
from mj_envs.utils.paths_utils import plot as plotnsave_paths
import numpy as np
import pickle
from mj_envs.utils.quat_math import mat2euler, quat2euler
import cv2
import os
from mj_envs.utils import tensor_utils
import random 

BEGIN_GRASP_THRESH = 0.08
SIM_BEGIN_GRASP_THRESH = 0.06
SIM_BEGIN_DESCENT_THRESH = 0.05
REAL_BEGIN_DESCENT_THRESH = 0.08
SIM_ALIGN_HEIGHT = 1.075
REAL_ALIGN_HEIGHT = 1.075
SIM_GRIPPER_FULL_OPEN = 0.04
SIM_GRIPPER_FULL_CLOSE = 0.00
REAL_GRIPPER_FULL_OPEN = 0.00
REAL_GRIPPER_FULL_CLOSE = 0.04
GRIPPER_BUFF_N = 8
GRIPPER_CLOSE_THRESH = 1e-8
MOVE_THRESH = 0.001

class HeuristicPolicy():
    def __init__(self, env, seed):
        self.env = env
        self.gripper_buff = SIM_GRIPPER_FULL_OPEN*np.ones((GRIPPER_BUFF_N,2))
        self.yaw = 0.0
        np.random.seed(seed)
        
    def get_action(self, obs):

        # TODO Change obsvec2dict to handle obs vectors with single dim
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        action = np.concatenate([obs_dict['grasp_pos'][0,0,:], [np.pi,0.0,self.yaw], obs_dict['qp'][0,0,7:9]])
        
        if np.linalg.norm(obs_dict['object_err'][0,0,:]) > SIM_BEGIN_GRASP_THRESH:
            # Reset gripper buff
            self.gripper_buff = SIM_GRIPPER_FULL_OPEN*np.ones((GRIPPER_BUFF_N,2))

            # Object not yet within gripper
            if np.linalg.norm(obs_dict['object_err'][0,0,:2]) > SIM_BEGIN_DESCENT_THRESH:

                #self.yaw = np.random.uniform(-3.14, 0.0)
                #print(quat2euler(np.concatenate([[self.env.sim_obsd.data.qpos[15]],self.env.sim_obsd.data.qpos[12:15]])))
                self.yaw = quat2euler(np.concatenate([[self.env.sim_obsd.data.qpos[15]],self.env.sim_obsd.data.qpos[12:15]]))[0]
                cur_yaw = mat2euler(self.env.sim.data.site_xmat[self.env.grasp_sid].reshape(3,3).transpose())[2]

                if np.abs(self.yaw-cur_yaw) > np.abs(self.yaw-np.pi-cur_yaw):
                    self.yaw -= np.pi
                # Gripper not yet aligned with object (also open gripper)
                action[2] = SIM_ALIGN_HEIGHT
                action[:2] += obs_dict['object_err'][0,0,0:2]
                action[6:8] = SIM_GRIPPER_FULL_OPEN
        
            else:
                # Gripper aligned with object, go down towards it (also open gripper)
                action[:3] += obs_dict['object_err'][0,0,0:3]
                action[6:8] = SIM_GRIPPER_FULL_OPEN
        else:
            # Close gripper, move to target once gripper has had chance to close
            for i in range(self.gripper_buff.shape[0]-1):
                self.gripper_buff[i] = self.gripper_buff[i+1]
            self.gripper_buff[-1] = obs_dict['qp'][0,0,7:9]

            if np.all(np.linalg.norm(self.gripper_buff - SIM_GRIPPER_FULL_OPEN, axis=1) > 1e-4):
                # Move to target
                action[:3] += obs_dict['target_err'][0,0,0:3]
            else:
                action[:3] += obs_dict['object_err'][0,0,0:3]

            action[6:8] = SIM_GRIPPER_FULL_CLOSE

        action = np.clip(action, self.env.pos_limit_low, self.env.pos_limit_high)
    
        cur_rot = mat2euler(self.env.sim_obsd.data.site_xmat[self.env.grasp_sid].reshape(3,3).transpose())
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot+2*np.pi-action[3:6])] += 2*np.pi
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot-2*np.pi-action[3:6])] -= 2*np.pi
        cur_pos = np.concatenate([self.env.sim_obsd.data.site_xpos[self.env.grasp_sid],
                                    cur_rot,
                                    [self.env.sim_obsd.data.qpos[7],0]])
        action = np.clip(action, cur_pos-self.env.eef_vel_limit, cur_pos+self.env.eef_vel_limit)

        # Normalize action to be between -1 and 1
        action = 2*(((action - self.env.pos_limit_low) / (self.env.pos_limit_high - self.env.pos_limit_low)) - 0.5)
        
        return action, {'evaluation': action}

class HeuristicPolicyReal():
    def __init__(self, 
                 env, 
                 seed, 
                 random_grasp_prob=0.0, 
                 output_dir=None, 
                 max_gripper_open=0.0002, 
                 min_gripper_closed=0.8228,
                 obj_pos_low=[-0.25,0.368,0.91],
                 obj_pos_high=[0.25, 0.72, 0.91],
                 pix_from_left = -43,
                 pix_from_top = 42,
                 x_dist_from_center = -0.5334,
                 y_dist_from_base = 0.72,
                 y_scale = -0.00342565789, 
                 x_scale = 0.00339745222,
                 mask_start_x = 32,
                 mask_end_x = 197,
                 mask_start_y = 41,
                 mask_end_y = 154):
        self.env = env

        self.yaw = 0.0
        self.stage = 0
        self.last_t = 0.0
        self.last_qp = None
        np.random.seed(seed)

        self.random_grasp_prob = random_grasp_prob
        self.output_dir = output_dir
        self.max_gripper_open = max_gripper_open
        self.min_gripper_closed = min_gripper_closed
        self.obj_pos_low = obj_pos_low
        self.obj_pos_high = obj_pos_high

        self.pix_from_left = pix_from_left
        self.pix_from_top = pix_from_top
        self.x_dist_from_center = x_dist_from_center
        self.y_dist_from_base = y_dist_from_base
        self.y_scale = y_scale
        self.x_scale = x_scale
        self.mask_start_x = mask_start_x
        self.mask_end_x = mask_end_x
        self.mask_start_y = mask_start_y
        self.mask_end_y = mask_end_y

        self.grasp_centers = None
        self.filtered_boxes = None
        self.img_masked = None

    # Adapted from here: https://github.com/fairinternal/robopen/blob/bb856233b3d8df89698ef80bf3c63ceab445b4aa/closed_loop_perception/robodev/policy/predict_grasp_release.py
    def update_grasps(self, img, out_dir=None):
        if out_dir is not None and not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        bin_mask = np.zeros(img.shape, dtype=np.uint8)

        bin_mask[self.mask_start_y:self.mask_end_y, self.mask_start_x:self.mask_end_x, :] = 255
        self.img_masked = cv2.bitwise_and(img, bin_mask)
        #img_masked_fn = os.path.join(out_dir, out_name+'_masked.png')
        #cv2.imwrite(img_masked_fn, self.img_masked)

        gray_img = cv2.cvtColor(self.img_masked, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray_img, 
                                        255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        15,
                                        15)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)

        # first box is background
        boxes = boxes[1:]
        self.filtered_boxes = []
        rec_img = self.img_masked.copy()
        for x,y,w,h,pixels in boxes:
            if pixels > 15 and h > 4 and w > 4:#pixels < 1000 and h < 40 and w < 40 and h > 4 and w > 4:
                self.filtered_boxes.append((x,y,w,h))
                cv2.rectangle(rec_img, (x,y), (x+w, y+h), (255,0,0), 1)

        if out_dir is not None:
            rec_img_fn = os.path.join(out_dir,'masked_all_recs.png')
            cv2.imwrite(rec_img_fn, cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR))
        
        
        random.shuffle(self.filtered_boxes)


        #for x,y,w,h in self.filtered_boxes:
        #    cv2.rectangle(self.img_masked, (x,y), (x+w, y+h), (0,0,255), 1)

        #rec_img_fn = os.path.join(out_dir, out_name+'recs.png')
        #cv2.imwrite(rec_img_fn, self.img_masked)

        self.grasp_centers = []
        for x,y,w,h in self.filtered_boxes:
            grasp_x = self.x_scale*((x+(w/2.0)) - self.pix_from_left)+self.x_dist_from_center
            grasp_y = self.y_scale*((y+(h/2.0)) - self.pix_from_top) + self.y_dist_from_base
            if (grasp_x >= self.obj_pos_low[0] and grasp_x <= self.obj_pos_high[0] and
                grasp_y >= self.obj_pos_low[1] and grasp_y <= self.obj_pos_high[1]):
                self.grasp_centers.append((grasp_x,grasp_y))
        return self.grasp_centers, self.filtered_boxes, self.img_masked

    def do_rollout(self, horizon):
        obs = self.env.reset()

        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        open_qp = obs_dict['qp'][0,0,:9].copy()
        open_qp[7:9] = 0.0
        done = False        

        while(obs_dict['qp'][0,0,7] > self.max_gripper_open and not done):
            release_action = 2*(((open_qp - self.env.jnt_low)/(self.env.jnt_high-self.env.jnt_low))-0.5)
            obs, _, done, env_info = self.env.step(release_action)
            obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        if (np.random.rand() > self.random_grasp_prob and
            self.grasp_centers is not None and
            len(self.grasp_centers) > 0):
            real_obj_pos = np.array([self.grasp_centers[-1][0],self.grasp_centers[-1][1], 0.91])
            
            if self.output_dir is not None:

                rec_img = self.img_masked.copy()
                x,y,w,h = self.filtered_boxes[-1]
                cv2.rectangle(rec_img, (x,y), (x+w, y+h), (255,0,0), 1)

                rec_img_fn = os.path.join(self.output_dir,'masked_latest.png')
                cv2.imwrite(rec_img_fn, cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR))

            self.grasp_centers.pop()
            self.filtered_boxes.pop()
            print('Vision obj pos')
        else:
            real_obj_pos = np.random.uniform(low=self.obj_pos_low, high=self.obj_pos_high)
            print('Random obj pos')

        target_pos = np.array([0.0, 0.5, 1.1])
        self.env.set_real_obj_pos(real_obj_pos)
        self.env.set_target_pos(target_pos)
        print('Real obs pos {} target pos {}'.format(real_obj_pos, target_pos))        

        rand_yaw = np.random.uniform(low = -3.14, high = 0)
        self.yaw = rand_yaw

        # start rollouts
        observations=[]
        actions=[]
        rewards=[]
        env_infos = []

        obs = self.env.get_obs()

        self.env.squeeze_dims(self.env.rwd_dict)
        self.env.squeeze_dims(self.env.obs_dict)
        env_info = self.env.get_env_infos()

        observations.append(obs)
        env_infos.append(env_info)

        solved = False
        done = False
        t = 0
        ep_rwd = 0.0
        while t < horizon:# and not done:
            a = self.get_action(obs)[0]
            next_o, rwd, done, env_info = self.env.step(a)      
            #a = np.concatenate([env.last_ctrl, a])
            obs_dict = self.env.obsvec2obsdict(np.expand_dims(next_o, axis=(0,1)))
            solved = env_info['solved'] and obs_dict['qp'][0,0,7] > 0.001
            ep_rwd += rwd

            observations.append(next_o)
            actions.append(a)
            rewards.append(rwd)
            env_infos.append(env_info)
            obs = next_o
            t = t+1

        path = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        terminated=done
        )

        return observations[-1], path

    def get_action(self, obs):

        # TODO Change obsvec2dict to handle obs vectors with single dim
        obs_dict = self.env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        action = np.concatenate([obs_dict['grasp_pos'][0,0,:], [np.pi,0.0,self.yaw], [obs_dict['qp'][0,0,7],0]])

        # Figure out which stage we are in
        if self.last_t > self.env.sim.data.time:
            # Reset
            self.stage = 0
            self.last_qp = None
            #self.yaw = np.random.uniform(low = 0.0, high = 3.14)
        elif self.stage == 0: # Wait until aligned xy
            # Advance to next stage?
            if (np.linalg.norm(obs_dict['object_err'][0,0,:2]) < REAL_BEGIN_DESCENT_THRESH and
                (self.last_qp is not None and np.linalg.norm(obs_dict['qp'][0,0,:] - self.last_qp) < 0.01)):#MOVE_THRESH)):
                self.stage = 1
        elif self.stage == 1:# Wait until close pregrasp
            if (np.linalg.norm(obs_dict['object_err'][0,0,:3]) < BEGIN_GRASP_THRESH or
                (self.last_qp is not None and np.linalg.norm(obs_dict['qp'][0,0,:] - self.last_qp) < 0.01)):#MOVE_THRESH)):
                self.stage = 2
        elif self.stage == 2: # Wait until pregrasp has stabilized
            # Advance to next stage?
            if (self.last_qp is not None and np.linalg.norm(obs_dict['qp'][0,0,:] - self.last_qp) < 0.001):#MOVE_THRESH):
                self.stage = 3
        elif self.stage == 3: # Wait for gripper to start closing
            # Advance to next stage?
            if (self.last_qp is not None and obs_dict['qp'][0,0,7] > self.last_qp[7]):
                self.stage = 4
        elif self.stage == 4: # Wait for gripper to stop closing
            if (self.last_qp is not None and np.abs(self.last_qp[7] - obs_dict['qp'][0,0,7]) < GRIPPER_CLOSE_THRESH):
                self.stage = 5   

        self.last_t = self.env.sim.data.time #obs_dict['t'][0,0,0]
        self.last_qp = obs_dict['qp'][0,0,:]

        #print('Stage {}, t {}'.format(self.stage, self.last_t))
        if self.stage == 0: # Align in xy
            action[2] = REAL_ALIGN_HEIGHT
            #action[:2] += obs_dict['object_err'][0,0,0:2]
            action[:2] += 1.5*obs_dict['object_err'][0,0,0:2]
            action[6] = REAL_GRIPPER_FULL_OPEN
            action[7] = 0
        elif self.stage == 1 or self.stage == 2: # Move to pregrasp
            #action[:2] += 2*obs_dict['object_err'][0,0,0:2]
            action[:2] += 1.5*obs_dict['object_err'][0,0,0:2]
            vel_alpha = (obs_dict['grasp_pos'][0,0,1]-0.368)/(0.72-0.368)
            #vel_limit_low = -0.075*vel_alpha+(1-vel_alpha)*-0.035
            vel_limit_low = -0.15
            action[2] += max(min(obs_dict['object_err'][0,0,2],0.15),vel_limit_low)
            action[6] = REAL_GRIPPER_FULL_OPEN
            action[7] = 0
        elif self.stage == 3 or self.stage == 4: # Close gripper
            action[:3] += obs_dict['object_err'][0,0,0:3]
            action[6] = REAL_GRIPPER_FULL_CLOSE
            action[7] = 0
            #print('Grasp pos {}'.format(obs_dict['grasp_pos'][0,0,:]))
        elif self.stage == 5: # Move to target pose
            action[:3] += obs_dict['target_err'][0,0,0:3]
            action[6] = REAL_GRIPPER_FULL_CLOSE
            action[7] = 0

        action = np.clip(action, self.env.pos_limit_low, self.env.pos_limit_high)
        cur_rot = mat2euler(self.env.sim_obsd.data.site_xmat[self.env.grasp_sid].reshape(3,3).transpose())
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot+2*np.pi-action[3:6])] += 2*np.pi
        cur_rot[np.abs(cur_rot-action[3:6])>np.abs(cur_rot-2*np.pi-action[3:6])] -= 2*np.pi
        cur_pos = np.concatenate([self.env.sim_obsd.data.site_xpos[self.env.grasp_sid],
                                    cur_rot,
                                    [self.env.sim_obsd.data.qpos[7],0]])
        action = np.clip(action, cur_pos-self.env.eef_vel_limit, cur_pos+self.env.eef_vel_limit)


        # Normalize action to be between -1 and 1
        action = 2*(((action - self.env.pos_limit_low) / (self.env.pos_limit_high - self.env.pos_limit_low)) - 0.5)

        return action, {'evaluation': action}
    
    def set_yaw(self, yaw):
        self.yaw = yaw

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
