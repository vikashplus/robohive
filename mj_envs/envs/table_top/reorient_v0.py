import os
import numpy as np
import collections
from gym import utils, spaces
from mjrl.envs import mujoco_env
from mj_envs.utils.quatmath import euler2quat, quat2euler
from mj_envs.utils.obj_vec_dict import ObsVecDict
from mujoco_py import MjViewer

OBS_KEYS = ['ctrl_jnt', 'obj_pos', 'obj_vel', 'obj_rot', 'obj_des_rot', 'obj_err_rot']
RWD_KEYS = ['rot_align', 'bonus']
RWD_MODE = 'dense' # dense/ sparse

class ReorientEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, ObsVecDict):
    def __init__(self, *args, **kwargs):
        # get sim
        sim = mujoco_env.get_sim(kwargs['model_path'])
        self.args = args
        self.kwargs = kwargs
        # ids
        self.target_obj_bid = sim.model.body_name2id("target")
        self.obj_bid = sim.model.body_name2id('Object')
        self.ctrl1_bid = sim.model.body_name2id('controller1')
        self.obj_t_sid = sim.model.site_name2id('object_top')
        self.obj_b_sid = sim.model.site_name2id('object_bottom')
        self.tar_t_sid = sim.model.site_name2id('target_top')
        self.tar_b_sid = sim.model.site_name2id('target_bottom')
        self.tool_handle_id = sim.model.geom_name2id('h_handle')
        self.tool_neck_id = sim.model.geom_name2id('h_neck')
        self.tool_head_id = sim.model.geom_name2id('h_head')
        self.tar_handle_sid = sim.model.site_name2id('target_handle')
        self.tar_neck_sid = sim.model.site_name2id('target_neck')
        self.tar_head_sid = sim.model.site_name2id('target_head')
        self.tool_length = np.linalg.norm(sim.model.site_pos[self.obj_t_sid] - sim.model.site_pos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(sim.model.site_pos[self.tar_t_sid] - sim.model.site_pos[self.tar_b_sid])
        # scales
        self.act_mid = np.mean(sim.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(sim.model.actuator_ctrlrange[:,1]-sim.model.actuator_ctrlrange[:,0])

        # get env
        utils.EzPickle.__init__(self)
        ObsVecDict.__init__(self)
        self.obs_dict = {}
        self.rwd_dict = {}
        mujoco_env.MujocoEnv.__init__(self, sim=sim, frame_skip=5)
        self.action_space = spaces.Box(-1.0*np.ones_like(self.action_space.low), np.ones_like(self.action_space.high), dtype=np.float32)


    # step the simulation forward
    def step(self, a):
        # apply action and step
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a*self.act_rng
        self.do_simulation(a, self.frame_skip)

        # observation and rewards
        obs = self.get_obs()
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()
        return obs, env_info['rwd_'+RWD_MODE], bool(env_info['done']), env_info


    def get_obs(self):
        # qpos for hand, xpos for obj, xpos for target
        self.obs_dict['t'] = np.array([self.sim.data.time])
        #self.obs_dict['hand_jnt'] = self.data.qpos[:-6].copy()
        self.obs_dict['ctrl_jnt'] = self.data.qpos[:-1].copy()
        self.obs_dict['obj_pos'] = self.data.body_xpos[self.obj_bid].copy()
        #self.obs_dict['obj_des_pos'] = self.data.site_xpos[self.eps_ball_sid].ravel()
        self.obs_dict['obj_vel'] = self.data.qvel[-1:].copy()
        self.obs_dict['obj_rot'] = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.tool_length
        self.obs_dict['obj_des_rot'] = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        #self.obs_dict['obj_err_pos'] = self.obs_dict['obj_pos']-self.obs_dict['obj_des_pos']
        self.obs_dict['obj_err_rot'] = self.obs_dict['obj_rot']-self.obs_dict['obj_des_rot']

        t, obs = self.obsdict2obsvec(self.obs_dict, OBS_KEYS)
        return obs

    def calculate_cosine(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Calculates the cosine angle between two vectors.

        This computes cos(theta) = dot(v1, v2) / (norm(v1) * norm(v2))

        Args:
            vec1: The first vector. This can have a batch dimension.
            vec2: The second vector. This can have a batch dimension.

        Returns:
            The cosine angle between the two vectors, with the same batch dimension
            as the given vectors.
        """
        if np.shape(vec1) != np.shape(vec2):
            raise ValueError('{} must have the same shape as {}'.format(vec1, vec2))
        ndim = np.ndim(vec1)
        norm_product = (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
        zero_norms = norm_product == 0
        if np.any(zero_norms):
            if ndim>1:
                norm_product[zero_norms] = 1
            else:
                norm_product = 1
        # Return the batched dot product.
        return np.einsum('...i,...i', vec1, vec2) / norm_product

    def get_reward_dict(self, obs_dict):
        #pos_err = obs_dict['obj_err_pos']
        #pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = self.calculate_cosine(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
        #print("Rotation reward: ", rot_align)
        #dropped = obs_dict['obj_pos'][:,:,2] < 0.075

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            #('pos_align',   -1.0*pos_align),
            ('rot_align',   1.0*rot_align),
            #('drop',        -5.0*dropped),
            ('bonus',       10.0*(rot_align > 0.9) + 50.0*(rot_align > 0.95)),
            # Must keys
            ('sparse',      rot_align),
            ('solved',      (rot_align > 0.95)),
            ('done',        False),
        ))
        rwd_dict['dense'] = np.sum([rwd_dict[key] for key in RWD_KEYS], axis=0)
        return rwd_dict


    # use latest obs, rwds to get all info (be careful, information belongs to different timestamps)
    # Its getting called twice. Once in step and sampler calls it as well
    def get_env_infos(self):
        env_info = {
            'time': self.obs_dict['t'][()],
            'rwd_dense': self.rwd_dict['dense'][()],
            'rwd_sparse': self.rwd_dict['sparse'][()],
            'solved': self.rwd_dict['solved'][()],
            'done': self.rwd_dict['done'][()],
            'obs_dict': self.obs_dict,
            'rwd_dict': self.rwd_dict,
        }
        return env_info


    # compute vectorized rewards for paths
    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs_dict = self.obsvec2obsdict(paths["observations"])

        rwd_dict = self.get_reward_dict(obs_dict)

        rewards = rwd_dict[RWD_MODE]
        done = rwd_dict['done']
        # time align rewards. last step is redundant
        done[...,:-1] = done[...,1:]
        rewards[...,:-1] = rewards[...,1:]
        paths["done"] = done if done.shape[0] > 1 else done.ravel()
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths


    def truncate_paths(self, paths):
        hor = paths[0]['rewards'].shape[0]
        for path in paths:
            if path['done'][-1] == False:
                path['terminated'] = False
                terminated_idx = hor
            elif path['done'][0] == False:
                terminated_idx = sum(~path['done'])+1
                for key in path.keys():
                    path[key] = path[key][:terminated_idx+1, ...]
                path['terminated'] = True
        return paths


    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=self.kwargs['goal'][0][0], high=self.kwargs['goal'][0][1])
        desired_orien[1] = self.np_random.uniform(low=self.kwargs['goal'][1][0], high=self.kwargs['goal'][1][1])
        desired_orien[2] = self.np_random.uniform(low=self.kwargs['goal'][2][0], high=self.kwargs['goal'][2][1])
        #print("Desired Orientation: ", desired_orien[2])
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.model.body_pos[self.ctrl1_bid] = self.np_random.uniform(
                                                                        low=np.array([self.kwargs["ctrl1"][ind][0] for ind in range(3)]),
                                                                        high=np.array([self.kwargs["ctrl1"][ind][1] for ind in range(3)])
                                                                    )
        if self.kwargs['generate_rd_tools']:
            #print(self.sim.model.geom_type[handle_id],self.sim.model.geom_type[neck_id],self.sim.model.geom_type[head_id])
            self.sim.model.geom_type[self.tool_handle_id] = self.np_random.choice([3,5])
            self.sim.model.geom_type[self.tool_neck_id] = self.np_random.choice([3,5])
            self.sim.model.geom_type[self.tool_head_id] = self.np_random.choice([3,5])
            self.sim.model.geom_rgba[self.tool_handle_id] = self.np_random.uniform(low=np.array([0., 0., 0., 1.]), high=np.array([1., 1., 1., 1.]))
            self.sim.model.geom_rgba[self.tool_neck_id] = self.np_random.uniform(low=np.array([0., 0., 0., 1.]), high=np.array([1., 1., 1., 1.]))
            self.sim.model.geom_rgba[self.tool_head_id] = self.np_random.uniform(low=np.array([0., 0., 0., 1.]), high=np.array([1., 1., 1., 1.]))
            #print(self.sim.model.geom_size[self.tool_head_id],self.sim.model.geom_size[self.tool_neck_id],self.sim.model.geom_size[self.tool_handle_id])
            #[0.02 0.04 0.] [0.007 0.085 0.] [0.025 0.05  0.]
            self.sim.model.geom_size[self.tool_head_id] = self.np_random.uniform(low=np.array([.01, .03, .0]), high=np.array([.03, .05, .0]))
            self.sim.model.geom_size[self.tool_neck_id] = self.np_random.uniform(low=np.array([.002, .085, .0]), high=np.array([.012, .135, .0]))
            self.sim.model.geom_size[self.tool_handle_id] = self.np_random.uniform(low=np.array([.015, .04, .0]), high=np.array([.035, .06, .0]))
            self.copy_tool2tar()

        self.sim.forward()
        return self.get_obs()

    def copy_tool2tar(self):
        self.sim.model.site_type[self.tar_handle_sid] = self.sim.model.geom_type[self.tool_handle_id]
        self.sim.model.site_type[self.tar_neck_sid]  = self.sim.model.geom_type[self.tool_neck_id]
        self.sim.model.site_type[self.tar_head_sid]  = self.sim.model.geom_type[self.tool_head_id]

        self.sim.model.site_rgba[self.tar_handle_sid] = self.sim.model.geom_rgba[self.tool_handle_id]
        self.sim.model.site_rgba[self.tar_neck_sid]  = self.sim.model.geom_rgba[self.tool_neck_id]
        self.sim.model.site_rgba[self.tar_head_sid]  = self.sim.model.geom_rgba[self.tool_head_id]

        self.sim.model.site_size[self.tar_handle_sid] = self.sim.model.geom_size[self.tool_handle_id]
        self.sim.model.site_size[self.tar_neck_sid]  = self.sim.model.geom_size[self.tool_neck_id]
        self.sim.model.site_size[self.tar_head_sid]  = self.sim.model.geom_size[self.tool_head_id]

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)


    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()


    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0


    # evaluate paths and log metrics to logger
    def evaluate_success(self, paths, logger=None):
        num_success = 0
        num_paths = len(paths)
        horizon = self.spec.max_episode_steps # paths could have early termination

        # success if pen within 15 degrees of target for 5 steps
        for path in paths:
            if np.sum(path['env_infos']['solved']) > 5:
                num_success += 1
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
            rwd_dense = np.mean([np.sum(p['env_infos']['rwd_sparse'])/horizon for p in paths]) # return rwd/step
            logger.log_kv('rwd_sparse', rwd_sparse)
            logger.log_kv('rwd_dense', rwd_dense)
            logger.log_kv('success_rate', success_percentage)

        return success_percentage
