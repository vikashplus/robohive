import gym
import mj_envs #even if unused it is needed to register the environments
import click
import numpy as np
import pickle

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
either onscreen, or offscreen, or just rollout without rendering.\n
USAGE:\n
    $ python examine_env.py --env_name door-v0 \n
    $ python examine_env.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
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
@click.option('-n', '--num_episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-f', '--filename', type=str, default='newvideo', help=('The name to save the rendered video as.'))
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))

def main(env_name, policy, mode, seed, num_episodes, render, filename, camera_name):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)

    # resolve policy
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = rand_policy(env)
        mode = 'exploration'

    # examine policy
    env.examine_policy(
        policy=pi,
        horizon=env.spec.max_episode_steps,
        num_episodes=num_episodes,
        frame_size=(640,480),
        mode=mode,
        filename=filename,
        camera_name=camera_name,
        render=render)

if __name__ == '__main__':
    main()


