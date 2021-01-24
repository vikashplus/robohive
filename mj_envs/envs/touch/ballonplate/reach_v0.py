from mj_envs.envs.touch.ballonplate.base_v0 import BallOnPlateBaseV0
import numpy as np
import collections

class BallOnPlateReachEnvV0(BallOnPlateBaseV0):

    def __init__(self,
                obs_keys:list = ['plate', 'dplate', 'ball', 'dball', 'target_err'],
                rwd_keys:list = ['reach', 'bonus', 'penalty'],
                target_xy_range = np.array(([-.05, -.05], [.05, .05])),
                ball_xy_range = np.array(([-.05, -.05], [.05, .05])),
                **kwargs):
        self.target_xy_range = target_xy_range
        self.ball_xy_range = ball_xy_range
        super().__init__(obs_keys=obs_keys, rwd_keys=rwd_keys)

    def get_obs(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['plate'] = self.data.qpos[:2].copy()
        self.obs_dict['ball'] = self.data.qpos[2:5].copy()
        self.obs_dict['dplate'] = self.data.qvel[:2].copy()
        self.obs_dict['dball'] = self.data.qvel[2:5].copy()
        self.obs_dict['target'] = self.data.site_xpos[self.target_sid].copy()
        self.obs_dict['target_err'] = self.obs_dict['target'] - self.obs_dict['ball']
        self.obs_dict['touch'] = self.data.sensordata.copy()

        # update touch rendering
        for i in range(len(self.obs_dict['touch'])):
            site_id = self.model.sensor_objid[i]
            self.sim.model.site_rgba[site_id, 3] = 1 if self.obs_dict['touch'][i]>0 else 0.2
            self.sim.model.site_rgba[site_id, 0] = 2 if self.obs_dict['touch'][i]>0 else 0.2

        t_1, obs_1 = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        obs = np.concatenate([obs_1])

        # Add old frames
        # t_2, obs_2 = self.get_obsvec_from_cache(-2)
        # obs = np.concatenate([obs, obs_2])

        # t_3, obs_3 = self.get_obsvec_from_cache(-3)
        # obs = np.concatenate([obs, obs_3])
        return obs

    def get_reward_dict(self, obs_dict):
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        ball_height = obs_dict['ball'][:,:,2]

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*target_dist),
            ('bonus',   4.0*(target_dist<.015) + 4.0*(target_dist<.030)),
            ('penalty', -50.*(ball_height<0.05)),
            # Must keys
            ('sparse',  -1.0*target_dist),
            ('solved',  target_dist<.015),
            ('done',    ball_height<0.05),
        ))
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in self.rwd_keys], axis=0)
        return rwd_dict

    def reset(self):
        # generate targets
        self.sim.model.site_pos[self.target_sid][:2] = self.np_random.uniform(low=self.target_xy_range[0], high=self.target_xy_range[1])
        self.sim.forward()

        # generate object
        qp = self.init_qpos.copy()
        qp[2:4] = self.np_random.uniform(low=self.ball_xy_range[0], high=self.ball_xy_range[1])
        obs = super().reset_model(qp=qp)
        return obs


