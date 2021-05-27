from mj_envs.envs.biomechanics.base_v0 import BaseV0
import numpy as np
import collections
from mjrl.envs.mujoco_env import get_sim
from mj_envs.utils.quatmath import euler2quat
from mj_envs.utils.vectormath import calculate_cosine

class PenTwirlFixedEnvV0(BaseV0):

    def __init__(self,
                obs_keys:list = ['hand_jnt', 'obj_pos', 'obj_vel', 'obj_rot', 'obj_des_rot', 'obj_err_pos', 'obj_err_rot'],
                rwd_keys:list = ['pos_align', 'rot_align', 'act_reg', 'drop', 'bonus'],
                **kwargs):

        self.sim = get_sim(model_path=kwargs['model_path'])

        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')
        self.pen_length = np.linalg.norm(self.sim.model.site_pos[self.obj_t_sid] - self.sim.model.site_pos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.sim.model.site_pos[self.tar_t_sid] - self.sim.model.site_pos[self.tar_b_sid])

        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys, sim=self.sim, rwd_viz=False, **kwargs)

    def get_obs(self):
        # qpos for hand, xpos for obj, xpos for target
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_jnt'] = self.data.qpos[:-6].copy()
        self.obs_dict['obj_pos'] = self.data.body_xpos[self.obj_bid].copy()
        self.obs_dict['obj_des_pos'] = self.data.site_xpos[self.eps_ball_sid].ravel()
        self.obs_dict['obj_vel'] = self.data.qvel[-6:].copy()
        self.obs_dict['obj_rot'] = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        self.obs_dict['obj_des_rot'] = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        self.obs_dict['obj_err_pos'] = self.obs_dict['obj_pos']-self.obs_dict['obj_des_pos']
        self.obs_dict['obj_err_rot'] = self.obs_dict['obj_rot']-self.obs_dict['obj_des_rot']
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        pos_err = obs_dict['obj_err_pos']
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
        dropped = obs_dict['obj_pos'][:,:,2] < 0.075

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pos_align',   -1.0*pos_align),
            ('rot_align',   1.0*rot_align),
            ('act_reg',     -5*np.linalg.norm(self.obs_dict['act'], axis=-1)),
            ('drop',        -5.0*dropped),
            ('bonus',       10.0*(rot_align > 0.9)*(pos_align<0.075) + 50.0*(rot_align > 0.95)*(pos_align<0.075) ),
            # Must keys
            ('sparse',      -1.0*pos_align+rot_align),
            ('solved',      (rot_align > 0.95)*(~dropped)),
            ('done',        dropped),
        ))
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict


class PenTwirlRandomEnvV0(PenTwirlFixedEnvV0):

    def reset_model(self):
        # randomize target
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        obs = super().reset_model()
        return obs