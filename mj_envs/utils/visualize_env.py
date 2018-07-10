from os import environ 
environ["MKL_THREADING_LAYER"] = "GNU"

import mj_envs  
import click 
import os
import gym
import numpy as np

DESC = '''
Helper script to visualize a NPG policy.\n
USAGE:\n
    Visualizes an env\n
    python examine_env.py --env_name <name>\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env. Use 'python examine_policy --help' for instructions")
        return

    # load envs
    env = gym.make(env_name)
    # render with random command
    visualize_random_act(env)


def visualize_random_act(env, num_episodes=5):
    act_dim = env.env.action_space.shape[0]
    for ep in range(num_episodes):
        o = env.reset()
        d = False
        t = 0
        while t < env.spec.timestep_limit and d is False:
            env.env.mj_render()
            a = np.random.uniform(low=-.1, high=0.1, size=act_dim)
            o, r, d, _ = env.step(a)
            t = t+1


if __name__ == '__main__':
    main()
