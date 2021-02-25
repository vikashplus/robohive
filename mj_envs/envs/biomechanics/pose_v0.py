from mj_envs.envs.biomechanics.base_v0 import FingerBaseV0
import numpy as np
import collections
from mjrl.envs.mujoco_env import get_sim

class PoseEnvV0(FingerBaseV0):

    def __init__(self,
                model_path:str,
                target_jnt_range:dict,
                viz_site_targets:tuple,
                obs_keys:list = ['qpos', 'qvel', 'pose_err'],
                rwd_keys:list = ['pose', 'bonus', 'penalty'],
                **kwargs):

        # pre-fetch sim
        sim = get_sim(model_path=model_path)

        # resolve joint demands
        self.target_jnt_ids = []
        self.target_jnt_range = []
        for jnt_name, jnt_range in target_jnt_range.items():
            self.target_jnt_ids.append(sim.model.joint_name2id(jnt_name))
            self.target_jnt_range.append(jnt_range)
        self.target_jnt_range = np.array(self.target_jnt_range)
        self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)

        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys, sites=viz_site_targets, sim=sim, **kwargs)

    def get_target_pose(self):
        return self.np_random.uniform(high=self.target_jnt_range[:,0], low=self.target_jnt_range[:,1])

    def get_obs(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.data.qvel[:].copy()
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.data.act[:].copy()

        self.obs_dict['pose_err'] = self.target_jnt_value - self.obs_dict['qpos']
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 4*np.pi/2

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',    -1.*pose_dist),
            ('bonus',   4.0*(pose_dist<.35) + 4.0*(pose_dist<.5)),
            ('penalty', -50.*(pose_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<.35),
            ('done',    pose_dist>far_th),
        ))
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict

    def reset(self):
        # generate targets
        self.target_jnt_value = self.get_target_pose()
        # update finger-tip target viz
        self.sim.data.qpos[:] = self.target_jnt_value.copy()
        self.sim.forward()
        for isite in range(len(self.tip_sids)):
            self.sim.model.site_pos[self.target_sids[isite]] = self.sim.data.site_xpos[self.tip_sids[isite]].copy()
        self.sim.forward()

        # initialize
        jnt_init = self.np_random.uniform(high=self.model.jnt_range[:,1], low=self.model.jnt_range[:,0])
        obs = super().reset_model(qp=jnt_init)
        return obs


