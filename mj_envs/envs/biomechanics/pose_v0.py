import collections
import gym
import numpy as np

from mj_envs.envs.biomechanics.base_v0 import BaseV0
from mj_envs.envs.env_base import get_sim

class PoseEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path:str, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path)

        self._setup(**kwargs)


    def _setup(self,
               target_jnt_range:dict=None,
               target_jnt_value:list=None,
               reset_type="none",
               target_type="generate",
               frame_skip = 10,
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        self.reset_type = reset_type
        self.target_type = target_type

        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       **kwargs)


    def get_obs_vec(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        self.obs_dict['pose_err'] = self.target_jnt_value - self.obs_dict['qpos']
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        obs_dict['pose_err'] = self.target_jnt_value - obs_dict['qpos']
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 4*np.pi/2

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',    pose_dist),
            ('bonus',   (pose_dist<.35) + (pose_dist<.5)),
            ('penalty', (pose_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<.35),
            ('done',    pose_dist>far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target pose
    def get_target_pose(self):
        if self.target_type == "fixed":
            return self.target_jnt_value
        elif self.target_type == "generate":
            return self.np_random.uniform(high=self.target_jnt_range[:,0], low=self.target_jnt_range[:,1])
        else:
            raise TypeError("Unknown Target type: {}".format(self.target_type))

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
        if self.target_type == "generate":
            # use target_jnt_range to generate targets
            self.update_target(restore_sim=True)
        elif self.target_type == "switch":
            # switch between given target choices
            # TODO: Remove hard-coded numbers
            if self.target_jnt_value[0] != -0.145125:
                self.target_jnt_value = np.array([-0.145125, 0.92524251, 1.08978337, 1.39425813, -0.78286243, -0.77179383, -0.15042819, 0.64445902])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11000209, -0.01753063, 0.20817679])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.1825131, 0.07417956, 0.11407256])
                self.sim.forward()
            else:
                self.target_jnt_value = np.array([-0.12756566, 0.06741454, 1.51352705, 0.91777418, -0.63884237, 0.22452487, 0.42103326, 0.4139465])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11647777, -0.05180014, 0.19044284])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.17728016, 0.01489491, 0.17953786])
        elif self.target_type is "fixed":
            self.update_target(restore_sim=True)
        else:
            print("{} Target Type not found ".format(self.target_type))

        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = super().reset()
        elif self.reset_type == "random":
            # reset to random state
            jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
            obs = super().reset(reset_qpos=jnt_init)
        else:
            print("Reset Type not found")

        return obs