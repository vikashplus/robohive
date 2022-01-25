import gym
import mj_envs #even if unused it is needed to register the environments
import click
import numpy as np
import pickle
import time
import os

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
        return self.env.action_space.sample(), {'mode': 'random samples'}

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy_path', type=str, help='absolute path of the policy file', default=None)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-v', '--video_name', type=str, default='video', help=('The name to save the rendered video as'))
@click.option('-l', '--log_name', type=str, default=None, help=('The name to save the rollout logs as'))

def main(env_name, policy_path, mode, seed, num_episodes, render, camera_name, output_dir, video_name, log_name):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)

    # resolve policy
    if policy_path is not None:
        pi = pickle.load(open(policy_path, 'rb'))
        if output_dir == './': # overide the default
           output_dir = policy_path.split('.')[0]+'/'
    else:
        pi = rand_policy(env)
        mode = 'exploration'

    # resolve directory
    if (os.path.isdir(output_dir) == False) and (render=='offscreen' or log_name is not None):
        os.mkdir(output_dir)

    # examine policy
    paths = env.examine_policy(
        policy=pi,
        horizon=env.spec.max_episode_steps,
        num_episodes=num_episodes,
        frame_size=(640,480),
        mode=mode,
        output_dir=output_dir,
        filename=video_name,
        camera_name=camera_name,
        render=render)

    if log_name is not None:
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = output_dir + log_name + '{}.pickle'.format(time_stamp)
        pickle.dump(paths, open(file_name, 'wb'))
        print("saved ", file_name)

if __name__ == '__main__':
    main()


