import gym
import mj_envs #even if unused it is needed to register the environments
import click
import gym
import numpy as np
import pickle
import time as timer

DESC = '''
Helper script to visualize policy.\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_env.py --env_name door-v0 \n
    $ python visualize_env.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# Random policy
class rand_policy():
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return [self.env.action_space.sample()]


def local_visualize_policy_offscreen(env, policy, horizon=1000,
                                num_episodes=1,
                                frame_size=(640,480),
                                mode='exploration',
                                save_loc='./tmp/',
                                filename='newvid',
                                camera_name=None):
    import skvideo.io
    import os 

    if not os.path.exists(save_loc): os.makedirs(save_loc)

    for ep in range(num_episodes):
        print("Episode %d: rendering offline " % ep, end='', flush=True)
        o = env.reset()
        d = False
        t = 0
        arrs = []
        t0 = timer.time()
        while t < horizon and d is False:
            a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
            o, r, d, _ = env.step(a)
            t = t+1
            curr_frame = env.sim.render(width=frame_size[0], height=frame_size[1],
                                            mode='offscreen', camera_name=camera_name, device_id=0)
            arrs.append(curr_frame[::-1,:,:])
            print(t, end=', ', flush=True)
        file_name = save_loc + filename + str(ep) + ".mp4"
            
        skvideo.io.vwrite( file_name, np.asarray(arrs),outputdict={"-pix_fmt": "yuv420p"})
        print("saved", file_name)
        t1 = timer.time()
        print("time taken = %f"% (t1-t0))

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--episodes', type=int, help='number of episodes to visualize', default=10)

        
def main(env_name, policy, mode, seed, episodes):
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = rand_policy(env)
        mode = 'exploration'

    # render policy
    if mode == 'evaluation':
        env.visualize_policy(pi, num_episodes=episodes, horizon=env.spec.max_episode_steps, mode=mode)
    elif mode == 'save_video':
        local_visualize_policy_offscreen(env, pi, num_episodes=episodes, filename='newvid_'+env_name+'_')

    
if __name__ == '__main__':
    main()

    
