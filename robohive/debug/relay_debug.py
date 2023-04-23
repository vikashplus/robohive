import gym
from robohive.utils.paths_utils import plot as plotnsave_paths
import click
import numpy as np
import pickle
# import time
import time as timer
import os

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots
USAGE:\n
    $ python examine_env.py --env_name door-v0 \n
    $ python examine_env.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''
from robohive.envs.obj_vec_dict import ObsVecDict
from robohive.utils import tensor_utils
from robohive.robot.robot import Robot
from robohive.utils.prompt_utils import prompt, Prompt
import skvideo.io
from sys import platform
from robohive.physics.sim_scene import SimScene
from robohive.utils.quat_math import quat2euler

import adept_envs

def viewer(env,
           mode='initialize',
           filename='video',
           frame_size=(640, 480),
           camera_id=0,
           render=None):
    if render == 'onscreen':
        env.mj_render()

    elif render == 'offscreen':

        global render_buffer
        if mode == 'initialize':
            render_buffer = []
            mode = 'render'

        if mode == 'render':
            curr_frame = env.render(mode='rgb_array')
            render_buffer.append(curr_frame)

        if mode == 'save':
            skvideo.io.vwrite(filename, np.asarray(render_buffer))
            print("\noffscreen buffer saved", filename)

    elif render == 'None':
        pass

    else:
        print("unknown render: ", render)

def adept_pickle_replay(env, env_name, original_paths):
    is_robohive_env = 'FK' in env_name
    if type(original_paths) is dict:
        original_paths = [original_paths]
    FPS = 30
    filename='demo_rendering.mp4'
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    render = 'onscreen'
    for data in original_paths:
        init_qpos = data['init_qpos'].copy()
        init_qvel = data['init_qvel'].copy()
        if is_robohive_env: # 6dof
            init_qpos = data['init_qpos'][:-1].copy()
            init_qpos[-3:] = quat2euler(data['init_qpos'][-4:])
            init_qvel = data['init_qvel'].copy()

            env.reset(reset_qpos=init_qpos, reset_qvel=init_qvel)
        else:
            env.reset()
        env.sim.data.qpos[:] = init_qpos
        env.sim.data.qvel[:] = init_qvel
        env.sim.forward()

        # Since we changed the sim state, we need to propagate changes throughout the envs
        if is_robohive_env:
            obs = env.env.get_obs()
        else:
            obs = env.env._get_obs()
        viewer(env, mode='initialize', render=render)

        print("qpos before step: ", env.sim.data.qpos)
        for i_frame in range(data['actions'].shape[0] - 1):
            act = data['actions'][i_frame]
            if is_robohive_env:
                act = act/np.pi # fix: The velocity limits are different in two the envs (+-2 for Adept_env and +-2*pi for Robohive. un-normalization of actions need to be done respecting the limits)
                next_obs, reward, done, env_info = env.env.step(act, update_exteroception=True)
            else:
                next_obs, reward, done, env_info = env.env.step(act)
            print("qpos after step: ", env.sim.data.qpos)
            # import pdb; pdb.set_trace() # comment this out to play the full trajectory
            if i_frame % render_skip == 0:
                viewer(env, mode='render', render=render)
                print(i_frame, end=', ', flush=True)
        if render:
            viewer(env, mode='save', filename=filename, render=render)
    import pdb; pdb.set_trace()

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-d', '--data_path', type=str, help='absolute path of the policy file', default=None)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-rv', '--render_visuals', type=bool, default=False, help=('render the visual keys of the env, if present'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
def main(env_name, data_path, mode, seed, render, camera_name, output_dir, output_name, save_paths, plot_paths, render_visuals, env_args):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)

    original_paths = pickle.load(open(data_path, 'rb'))
    paths = adept_pickle_replay(
        env,
        env_name,
        original_paths=original_paths
    )


if __name__ == '__main__':
    main()