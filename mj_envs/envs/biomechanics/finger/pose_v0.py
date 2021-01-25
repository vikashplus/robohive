from mj_envs.envs.biomechanics.finger.base_v0 import FingerBaseV0
import numpy as np
import collections

class PoseEnvV0(FingerBaseV0):

    def __init__(self,
                obs_keys:list = ['qpos', 'qvel', 'pose_err', 'act'],
                rwd_keys:list = ['pose', 'bonus', 'penalty'],
                pose_target_jnt_range = np.array(([-.1, -.1, .7, .7], [.1, .1, .8, .8])),
                **kwargs):
        self.pose_target_jnt_range = pose_target_jnt_range
        self.pose_target_jnt = np.mean(self.pose_target_jnt_range, axis=0)
        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys, **kwargs)

    def get_target_pose(self):

        return self.np_random.uniform(high=self.pose_target_jnt_range[0], low=self.pose_target_jnt_range[1])

    def get_obs(self):
        # import ipdb; ipdb.set_trace()
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.data.qvel[:].copy()
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.data.act[:].copy()
        self.obs_dict['pose_err'] = self.pose_target_jnt - self.obs_dict['qpos']
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 4*np.pi/2

        # print(pose_dist)
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',   -1.*pose_dist),
            ('bonus',   4.0*(pose_dist<.35) + 4.0*(pose_dist<.5)),
            ('penalty', -50.*(pose_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<.35),
            ('done',    pose_dist>far_th),
        ))
        # print(pose_dist, rwd_dict['solved'])

        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict

    def reset(self):
        # generate targets
        self.pose_target_jnt = self.get_target_pose()
        # update finger-tip target viz
        self.sim.data.qpos[:] = self.pose_target_jnt.copy()
        self.sim.forward()
        self.sim.model.site_pos[self.target_sid] = self.sim.data.site_xpos[self.tip_sid].copy()
        self.sim.forward()

        # initialize
        jnt_init = self.np_random.uniform(high=self.model.jnt_range[:,1], low=self.model.jnt_range[:,0])
        obs = super().reset_model(qp=jnt_init)
        return obs


