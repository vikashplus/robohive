from mj_envs.envs.biomechanics.base_v0 import BaseV0
import numpy as np
import collections
from mjrl.envs.mujoco_env import get_sim

class PoseEnvV0(BaseV0):

    def __init__(self,
                model_path:str,
                target_jnt_range:dict,
                viz_site_targets:tuple,
                reset_type = "init",  # none; init; random
                target_type = "generate", # generate; switch
                obs_keys:list = ['qpos', 'qvel', 'pose_err'],
                rwd_keys:list = ['pose', 'bonus', 'penalty'],
                **kwargs):

        self.reset_type = reset_type
        self.target_type = target_type

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

    # generate a valid target pose
    def get_target_pose(self):
        return self.np_random.uniform(high=self.target_jnt_range[:,0], low=self.target_jnt_range[:,1])

    # update sim with a new target pose
    def update_target(self, restore_sim=False):
        if restore_sim:
            qpos = self.sim.data.qpos[:].copy()
            qvel = self.sim.data.qvel[:].copy()
        # generate targets
        self.target_jnt_value = self.get_target_pose()
        # update finger-tip target viz
        self.sim.data.qpos[:] = self.target_jnt_value.copy()
        self.sim.forward()
        for isite in range(len(self.tip_sids)):
            self.sim.model.site_pos[self.target_sids[isite]] = self.sim.data.site_xpos[self.tip_sids[isite]].copy()
        if restore_sim:
            self.sim.data.qpos[:] = qpos[:]
            self.sim.data.qvel[:] = qvel[:]
        self.sim.forward()

    # reset_type = none; init; random
    # target_type = generate; switch
    def reset(self):

        # update target
        if self.target_type is "generate":
            # use target_jnt_range to generate targets
            self.update_target(restore_sim=True)
        elif self.target_type is "switch":
            # switch between given target choices
            if self.target_jnt_value[0] != -0.145125:
                self.target_jnt_value = np.array([-0.145125, 0.92524251, 1.08978337, 1.39425813, -0.78286243, -0.77179383, -0.15042819, 0.64445902])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11000209, -0.01753063, 0.20817679])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.1825131, 0.07417956, 0.11407256])
                self.sim.forward()
            else:
                self.target_jnt_value = np.array([-0.12756566, 0.06741454, 1.51352705, 0.91777418, -0.63884237, 0.22452487, 0.42103326, 0.4139465])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11647777, -0.05180014, 0.19044284])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.17728016, 0.01489491, 0.17953786])
        else:
            print("Target Type not found")

        # update init state
        if self.reset_type is "none" or self.reset_type is None:
            # no reset; use last state
            obs = self.get_obs()
        elif self.reset_type is "init":
            # reset to init state
            obs = super().reset_model()
        elif self.reset_type is "random":
            # reset to random state
            jnt_init = self.np_random.uniform(high=self.model.jnt_range[:,1], low=self.model.jnt_range[:,0])
            obs = super().reset_model(qp=jnt_init)
        else:
            print("Reset Type not found")

        return obs