import numpy as np
import gym
from abc import ABC
import numpy as np
from operator import itemgetter

class GymPixelWrapper(gym.Env, ABC):
    def __init__(
                    self,
                    env,
                    cameras,
                    hybrid_state=False,
                    height=100, width=100,
                    channels_first=False,
                    device_id=0
                ):
        """
        Construct a new 'GymPixelWrapper' object.

        :param env: Gym env object
        :param cameras: List of cameras to render image from
        :param hybrid_state: Return specific state values along with image
        :param height: height of the image
        :param width: width of the image
        :param channels_first: If True returns (num_cam, 3, height, width) else (num_cam, height, width, 3)
        :param device_id: device_id of the gpu
        """

        self.env = env
        self.cameras = cameras
        self.hybrid_state = hybrid_state
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.device_id = device_id
        self.env_kwargs = {'cameras' : cameras, 'hybrid_state':hybrid_state, 'height' : height, 'width':width, 'channels_first':channels_first}

        self.env_id = self.env.unwrapped.spec.id
        shape = [len(cameras), 3, height, width] if channels_first else [len(cameras), height, width, 3]
        self._observation_space = gym.spaces.Box(  # Not correct observation space if hybrid_state=True.
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self._action_space = self.env.action_space
        self.sim = self.env.sim
        self.metadata = self.env.metadata
        self.spec = getattr(self.env, 'spec', None)

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps
    @property
    def horizon(self):
        return self.env._max_episode_steps
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def action_space(self):
        return self._action_space
    @property
    def unwrapped(self):
        return self.env.unwrapped
    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)
    def __repr__(self):
        return str(self)
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def set_env_state(self, state):
        return self.env.set_env_state(state)
    def get_env_state(self):
        return self.env.get_env_state()

    def get_hybrid_state(self, env_state=None): # TODO Rutav 01/09: Change env specific behavior?
        if not env_state :
            env_state = [self.get_env_state()]
        qp = np.asarray(list(map(itemgetter('qpos'), env_state))) # Handles a list of env_states
        if self.env_id == 'pen-v0':
            qp = qp[:,:-6]
        elif self.env_id == 'door-v0':
            qp = qp[:,4:-2]
        elif self.env_id == 'hammer-v0':
            qp = qp[:,2:-7]
        elif self.env_id == 'relocate-v0':
            qp = qp[:,6:-6]
        else:
            raise NotImplementedError
        return qp

    def get_pixel_obs(
                          self,
                          height,
                          width,
                          cameras=[],
                          channels_first=False,
                     ):
        imgs = []
        imgs = np.zeros(self._observation_space.shape, dtype=np.uint8)
        for ind, cam in enumerate(cameras) :
            img = self.sim.render(width=width, height=height, mode='offscreen', camera_name=cam, device_id=self.device_id)
            img = img[::-1, :, : ] # Image has to be flipped
            if channels_first :
                img = img.transpose((2, 0, 1))
            imgs[ind, :, :, :] = img
        return imgs

    def get_obs(self, state=None):
        imgs = self.get_pixel_obs(
                                    height=self.height, width=self.width,
                                    cameras=self.cameras,
                                    channels_first=self.channels_first
                                )
        if self.hybrid_state:
            return [imgs, self.get_hybrid_state()]
        else:
            return [imgs]

    def get_env_infos(self):
        return self.env.get_env_infos()

    def set_seed(self, seed):
        try:
            return self.env.seed(seed)
        except:
            return self.env.set_seed(seed)

    def reset(self):
        obs = self.env.reset()
        obs = self.get_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, env_info = self.env.step(action)
        obs = self.get_obs(obs)
        return obs, reward, done, env_info
