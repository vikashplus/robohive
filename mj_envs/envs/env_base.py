import gym
import numpy as np
import os
import time as timer

from mj_envs.utils.obj_vec_dict import ObsVecDict
from mj_envs.robot.robot import Robot
from os import path
from xml.dom import InvalidStateErr

# TODO
# remove rwd_mode
# convet obs_keys to obs_keys_wt

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer, load_model_from_xml, ignore_mujoco_warnings
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def get_sim(model_path:str=None, model_xmlstr=None):
    """
    Get sim using model_path or model_xmlstr.
    """
    if model_path:
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = load_model_from_path(fullpath)
    elif model_xmlstr:
        model = load_model_from_xml(model_xmlstr)
    else:
        raise TypeError("Both model_path and model_xmlstr can't be None")

    return MjSim(model)

class MujocoEnv(gym.Env, gym.utils.EzPickle, ObsVecDict):
    """
    Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path):
        self.sim = get_sim(model_path)
        self.sim_obsd = get_sim(model_path)
        self.real_step = True
        self.env_timestep = 0
        ObsVecDict.__init__(self)

    def _setup(self,
               obs_keys,
               weighted_reward_keys,
               reward_mode = "dense",
               frame_skip = 1,
               normalize_act = True,
               obs_range = (-10, 10),
               seed = None,
               rwd_viz = False,
               **kwargs,
        ):

        if self.sim is None or self.sim_obsd is None:
            raise InvalidStateErr("sim and sim_obsd must be instantiated for setup to run")

        # seed the random number generator
        self.input_seed = None
        self.seed(seed)
        self.mujoco_render_frames = False
        self.rwd_viz = rwd_viz

        # resolve robot config
        self.robot = Robot(mj_sim=self.sim,
                           random_generator=self.np_random,
                           **kwargs)

        #resolve action space
        self.frame_skip = frame_skip
        self.normalize_act = normalize_act
        act_low = -np.ones(self.sim.model.nu) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,0].copy()
        act_high = np.ones(self.sim.model.nu) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,1].copy()
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # resolve rewards
        self.rwd_dict = {}
        self.rwd_mode = reward_mode
        self.rwd_keys_wt = weighted_reward_keys

        # resolve obs
        self.env_state_dict = self.get_env_state().copy()
        self.obs_dict = {}
        self.obs_keys = obs_keys
        observation, _reward, done, _info = self.step(np.zeros(self.sim.model.nu))
        assert not done, "Check initialization. Simulation starts in a done state."
        self.obs_dim = observation.size
        self.observation_space = gym.spaces.Box(obs_range[0]*np.ones(self.obs_dim), obs_range[1]*np.ones(self.obs_dim), dtype=np.float32)

        # resolve initial state
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.init_qpos = self.sim.data.qpos.ravel().copy() # has issues with initial jump during reset
        # self.init_qpos = np.mean(self.sim.model.actuator_ctrlrange, axis=1) if self.normalize_act else self.sim.data.qpos.ravel().copy() # has issues when nq!=nu
        # self.init_qpos[self.sim.model.jnt_dofadr] = np.mean(self.sim.model.jnt_range, axis=1) if self.normalize_act else self.sim.data.qpos.ravel().copy()
        if self.normalize_act:
            linear_jnt_qposids = self.sim.model.jnt_qposadr[self.sim.model.jnt_type>1] #hinge and slides
            linear_jnt_ids = self.sim.model.jnt_type>1
            self.init_qpos[linear_jnt_qposids] = np.mean(self.sim.model.jnt_range[linear_jnt_ids], axis=1)

        return

    def step(self, a):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        """
        a = np.clip(a, self.action_space.low, self.action_space.high)
        self.last_ctrl = self.robot.step(ctrl_desired=a,
                                        ctrl_normalized=self.normalize_act,
                                        step_duration=self.dt,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # increment environment timestep
        self.env_timestep += 1

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # update the env state tracking
        self.env_state_dict = self.get_env_state().copy()

        # returns obs(t+1), rew(t), done(t), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info


    def get_obs(self):
        """
        Get observations from the environemnt.
        Uses robot to get sensors, reconstructs the sim and recovers the sensors.
        """
        # get sensor data from robot
        sen = self.robot.get_sensors()

        # reconstruct (partially) observed-sim using (noisy) sensor data
        self.robot.sensor2sim(sen, self.sim_obsd)

        # get obs_dict using the observed information
        self.obs_dict = self.get_obs_dict(self.sim_obsd)

        # recoved observation vector from the obs_dict
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs


    # VIK??? Its getting called twice. Once in step and sampler calls it as well
    def get_env_infos(self):
        """
        Get information about the environment.
        - Essential keys are added below. Users can add more keys
        - Requires necessary keys (dense, sparse, solved, done) in rwd_dict to be populated
        - Note that entries belongs to different MDP steps
        """
        env_info = {
            'time': self.obs_dict['t'][()],             # MDP(t)
            'rwd_dense': self.rwd_dict['dense'][()],    # MDP(t-1)
            'rwd_sparse': self.rwd_dict['sparse'][()],  # MDP(t-1)
            'solved': self.rwd_dict['solved'][()],      # MDP(t-1)
            'done': self.rwd_dict['done'][()],          # MDP(t-1)
            'obs_dict': self.obs_dict,                  # MDP(t)
            'rwd_dict': self.rwd_dict,                  # MDP(t-1)
            'env_state': self.env_state_dict.copy()     # MDP(t-1)
        }
        return env_info


    # Methods on paths =======================================================

    def compute_path_rewards(self, paths):
        """
        Compute vectorized rewards for paths and check for done conditions
        path has two keys: observations and actions
        path["observations"] : (num_traj, horizon, obs_dim)
        path["rewards"] should have shape (num_traj, horizon)
        """
        obs_dict = self.obsvec2obsdict(paths["observations"])
        rwd_dict = self.get_reward_dict(obs_dict)

        rewards = rwd_dict[self.rwd_mode]
        done = rwd_dict['done']
        # time align rewards. last step is redundant
        done[...,:-1] = done[...,1:]
        rewards[...,:-1] = rewards[...,1:]
        paths["done"] = done if done.shape[0] > 1 else done.ravel()
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths


    def truncate_paths(self, paths):
        """
        truncate paths as per done condition
        """
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


    def evaluate_success(self, paths, logger=None, successful_steps=5):
        """
        Evaluate paths and log metrics to logger
        """
        num_success = 0
        num_paths = len(paths)

        # Record success if solved for provided successful_steps
        for path in paths:
            if np.sum(path['env_infos']['solved'] * 1.0) > successful_steps:
                # sum of truth values may not work correctly if dtype=object, need to * 1.0
                num_success += 1
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
            rwd_dense = np.mean([np.sum(p['env_infos']['rwd_dense'])/self.horizon for p in paths]) # return rwd/step
            logger.log_kv('rwd_sparse', rwd_sparse)
            logger.log_kv('rwd_dense', rwd_dense)
            logger.log_kv('success_percentage', success_percentage)

        return success_percentage


    def seed(self, seed=None):
        """
        Set random number seed
        """
        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def get_input_seed(self):
        return self.input_seed


    def reset(self, reset_qpos=None, reset_qvel=None):
        """
        Reset the environment
        Default implemention provided. Override if env needs custom reset
        """
        qpos = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qvel = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(qpos, qvel)
        self.env_timestep = 0
        self.env_state_dict = self.get_env_state().copy()
        return self.get_obs()


    @property
    def _step(self, a):
        return self.step(a)


    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip


    @property
    def horizon(self):
        return self.spec.max_episode_steps # paths could have early termination before horizon


    # state utilities ========================================================

    def set_state(self, qpos, qvel):
        """
        Set MuJoCo sim state
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def get_env_state(self):
        """
        Get full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        mocap_pos = self.sim.data.mocap_pos.copy() if self.sim.model.nmocap>0 else None
        mocap_quat = self.sim.data.mocap_quat.copy() if self.sim.model.nmocap>0 else None
        site_pos = self.sim.model.site_pos[:].copy() if self.sim.model.nsite>0 else None
        body_pos = self.sim.model.body_pos[:].copy()
        env_timestep = self.env_timestep
        return dict(qpos=qp,
                    qvel=qv,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                    site_pos=site_pos,
                    site_quat=self.sim.model.site_quat[:].copy(),
                    body_pos=body_pos,
                    body_quat=self.sim.model.body_quat[:].copy(),
                    env_timestep=self.env_timestep)


    def set_env_state(self, state_dict):
        """
        Set full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.sim.model.site_pos[:] = state_dict['site_pos']
        self.sim.model.site_quat[:] = state_dict['site_quat']
        self.sim.model.body_pos[:] = state_dict['body_pos']
        self.sim.model.body_quat[:] = state_dict['body_quat']
        self.sim.forward()

    # def state_vector(self):
    #     state = self.sim.get_state()
    #     return np.concatenate([
    #         state.qpos.flat, state.qvel.flat])


    # Vizualization utilities ================================================

    def mj_render(self):
        try:
            self.viewer.render()
        except:
            self.viewer = MjViewer(self.sim)
            self.viewer._run_speed = 0.5
            self.viewer.cam.elevation = -30
            self.viewer.cam.azimuth = 90
            self.viewer.cam.distance = 2.5
            # self.viewer.lookat = np.array([-0.15602934,  0.32243594,  0.70929817])
            #self.viewer._run_speed /= self.frame_skip
            self.viewer_setup()
            self.viewer.render()


    def update_camera(self, camera=None, distance=None, azimuth=None, elevation=None, lookat=None):
        """
        Updates the given camera to move to the provided settings.
        """
        if not camera:
            if not self.viewer:
                return
            else:
                camera = self.viewer
        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation
        if lookat is not None:
            camera.lookat[:] = lookat

    # def render(self, *args, **kwargs):
    #     pass
    #     #return self.mj_render()

    # def _get_viewer(self):
    #     pass
    #     #return None


    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            score = 0.0
            while t < horizon and d is False:
                # o = self._get_obs()
                # import ipdb; ipdb.set_trace()

                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                score = score + r
            print("Total episode reward = %f" % score)
        self.mujoco_render_frames = False


    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640,480),
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            o = self.reset()
            d = False
            t = 0
            arrs = []
            t0 = timer.time()
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1,:,:])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + str(ep) + ".mp4"
            skvideo.io.vwrite( file_name, np.asarray(arrs))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f"% (t1-t0))


    # methods to override ====================================================

    def get_obs_dict(self, sim):
        """
        Get observation dictionary
        Implement this in each subclass.
        """
        raise NotImplementedError


    def get_reward_dict(self, obs_dict):
        """
        Compute rewards dictionary
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function. Customize your viewer here
        """
        pass
