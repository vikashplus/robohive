import gym
import mj_envs
import click
import gym
import numpy as np
import pickle

DESC = '''
Helper script to visualize policy.\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_policy.py --env_name door-v0 \n
    $ python visualize_policy.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# Random policy
class rand_policy():
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return [self.env.action_space.sample()]

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--episodes', type=int, help='number of episodes to visualize', default=10)

def main(env_name, policy, mode, seed, episodes):
    env = gym.make(env_name)
    env.seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = rand_policy(env)
        mode = 'exploration'

    # render policy
    env.visualize_policy(pi, num_episodes=episodes, horizon=env.spec.max_episode_steps, mode=mode)

if __name__ == '__main__':
    main()