import gym
import click
import time
from termcolor import colored
import mj_envs
import multiprocessing as mp
from mj_envs.envs.pixel_wrapper import GymPixelWrapper

from PIL import Image

DESC = ""
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', default="mjrl_swimmer-v0")
@click.option('-e','--episodes', type=int, help='Number of episodes', default=10)
@click.option('-c','--camera', type=str, help='Camera name', default='cam0')
@click.option('-p','--policy', type=str, help='Path to policy', default=None)
def main(env_name, episodes, camera, policy):
    env = gym.make(env_name, device_id=0)
    #env = GymPixelWrapper(env, [camera], hybrid_state=True)
    mode = 'exploration'
    width = 84
    height = 84
    total_rendering_time = 0.0
    total_steps = 0
    obs = env.reset()
    for ep in range(episodes):

        o = env.reset()
        d = False
        t = 0
        score = 0.0
        while d is False:
            start_time = time.time()
            o, r, d, _ = env.step(env.action_space.sample())
            total_rendering_time += time.time() - start_time
            t = t+1
            total_steps += 1
            score = score + r

    avg_rendering_time = total_rendering_time / total_steps
    print(colored("Average time to take one step : {}".format(avg_rendering_time), "red"))
    print(colored("Frequency : {} steps/second".format(1/avg_rendering_time), "red"))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
