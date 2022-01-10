import mj_envs
import gym

#env = gym.make('reorient-fixed-v0')
env = gym.make('reorient-random-v0')
obs = env.reset()
act = env.action_space.sample()
env.step(act)
