import gym
import mj_envs
import click
import gym
import numpy as np
import pickle
import os
from PIL import Image

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
@click.option('-c', '--cam_name', type=str, help='Camera name', default=None)

def main(env_name, policy, mode, seed, episodes, cam_name):
    env = gym.make(env_name)
    env.seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = rand_policy(env)
        mode = 'exploration'

    output_dir = "{}_".format(env_name) + "_" + cam_name + "/" 
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
 

    # render policy
    #env.visualize_policy(pi, num_episodes=episodes, horizon=env.spec.max_episode_steps, mode=mode)
    for ep in range(episodes):
        o = env.reset()
        d = False
        t = 0
        score = 0.0
        while t < env.horizon and d is False:
            # o = self._get_obs()
            # import ipdb; ipdb.set_trace()

            a = pi.get_action(o)[0] if mode == 'exploration' else pi.get_action(o)[1]['evaluation']

            img_arr = env.env.sim.render(width=640, height=480, mode='offscreen', camera_name=cam_name, device_id=0)
            img = Image.fromarray(img_arr[::-1,:,:])
            img.save(output_dir + f"img_ep{ep:05d}_t{t:05d}.jpeg")

            o, r, d, _ = env.step(a)
            t = t+1
            score = score + r
        print("Total episode reward = %f" % score)


if __name__ == '__main__':
    main()
