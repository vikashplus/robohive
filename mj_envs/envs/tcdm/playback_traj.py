import mj_envs
import gym
import time

env = gym.make('AdroitBananaPass-v0')
# env = gym.make('AdroitBananaFixed-v0')
# env = gym.make('AdroitBananaRandom-v0')

env.reset()
while env.playback():
    env.mj_render()
    time.sleep(.1)
env.close()
