DESC = '''
Helper script to record/examine a rollout's openloop effects (render/ playback/ recover) on an environment\n
  > Examine options:\n
    - Record:   Record an execution. (Useful for kinesthetic demonstrations on hardware)\n
    - Render:   Render back the execution. (sim.forward)\n
    - Playback: Playback the rollout action sequence in openloop (sim.step(a))\n
    - Recover:  Playback actions recovered from the observations \n
  > Render options\n
    - either onscreen, or offscreen, or just rollout without rendering.\n
  > Save options:\n
    - save resulting paths as pickle or as 2D plots\n
USAGE:\n
    $ python examine_rollout.py --env_name door-v0 \n
    $ python examine_rollout.py --env_name door-v0 --rollout_path my_rollouts.pickle --repeat 10 \n
'''

from robohive.utils.paths_utils import plot as plotnsave_paths
from robohive.logger.grouped_datasets import Trace
from robohive.utils import tensor_utils
import gym
import click
import numpy as np
import time
import os
import skvideo.io
from sys import platform


@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required=True)
@click.option('-p', '--rollout_path', type=str, help='absolute path of the rollout', default=None)
@click.option('-f', '--rollout_format', type=click.Choice(['RoboHive', 'RoboSet']), help='Data format', default='RoboHive')
@click.option('-m', '--mode', type=click.Choice(['record', 'render', 'playback', 'recover']), help='How to examine rollout', default='playback')
@click.option('-h', '--horizon', type=int, help='Rollout horizon, when mode is record', default=-1)
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_repeat', type=int, help='number of repeats for the rollouts', default=1)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-fs', '--frame_size', type=tuple, default=(640,480), help=('Camera frame size for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-cp', '--compress_paths', type=bool, default=True, help=('compress paths. Remove obs and env_info/state keys'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-ns', '--noise_scale', type=float, default=0.0, help=('Noise amplitude in randians}"'))

def main(env_name, rollout_path, rollout_format, mode, horizon, seed, num_repeat, render, camera_name, frame_size, output_dir, output_name, save_paths, compress_paths, plot_paths, env_args, noise_scale):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)

    # Start a "trace" for recording rollouts
    if rollout_format=='RoboHive':
        from robohive.logger.grouped_datasets import Trace
    elif rollout_format=='RoboSet':
        from robohive.logger.roboset_logger import RoboSet_Trace as Trace
    else:
        raise TypeError("unknown rollout_format format")
    trace = Trace("Rollouts")

    # Load old traces as "paths"; none if record
    if mode == 'record':
        assert horizon>0, "Rollout horizon must be specified when recording rollout"
        assert output_name is not None, "Specify the name of the recording"
        if save_paths is False:
            print("Warning: Recording is not being saved. Enable save_paths=True to log the recorded path")
        paths = [None,]*num_repeat # Mark old traces as None
    else:
        assert rollout_path is not None, "Rollout path is required for mode:{} ".format(mode)
        if output_dir == './': # overide the default
            output_dir = os.path.dirname(rollout_path)
        if output_name is None: # default to the rollout name
            rollout_name = os.path.split(rollout_path)[-1]
            output_name, output_type = os.path.splitext(rollout_name)
        # file_name = os.path.join(output_dir, output_name+"_"+"-".join(cam_names))
        paths = Trace.load(rollout_path)

    # Resolve rendering
    if render == 'onscreen':
        env.env.mujoco_render_frames = True
    elif render =='offscreen':
        env.mujoco_render_frames = False
        frames = np.zeros((env.horizon+1, frame_size[1], frame_size[0], 3), dtype=np.uint8)
    elif render == None:
        env.mujoco_render_frames = False

    # Rollout paths
    for i_loop in range(num_repeat):

        # Rollout path
        print("Starting rollout loop:{}".format(i_loop))
        for path_name, path_data in paths.items():
            print(path_name, path_data)

            if rollout_format == "robohive":
                path_data = path_data
            elif rollout_format == "roboset":
                path_data = path_data['data']

            # initialize path -----------------------------
            ep_t0 = time.time()
            path_name+='-'+str(i_loop)
            print("Starting {} rollout".format(path_name))
            trace.create_group(path_name)

            # init: reset to starting state
            if path_data:
                # reset env
                if rollout_format=='RoboSet':
                    # reset to init state
                    reset_qpos = env.init_qpos.copy()
                    nq_arm = len(path_data['qp_arm'][0])
                    reset_qpos[:nq_arm] = path_data['qp_arm'][0]
                    nq_ee = len(path_data['qp_ee'][0])
                    reset_qpos[nq_arm:nq_arm+nq_ee] = path_data['qp_ee'][0] # assumption
                    env.reset(reset_qpos=reset_qpos)
                elif rollout_format=='RoboHive' and "state" in path_data['env_infos'].keys():
                    env.reset(reset_qpos=path_data['env_infos']['state']['qpos'][0], reset_qvel=path_data['env_infos']['state']['qvel'][0])
                else:
                    raise TypeError("Unknown path type")
                # recover env other initial state
                state_t = tensor_utils.split_tensor_dict_list(path_data['env_infos']['state'])
                env.env.set_env_state(state_t[0])
            else:
                env.reset()
            trace_horizon = horizon if mode=='record' else path_data['time'].shape[0]-1


            # Rollout path --------------------------------
            ep_rwd = 0.0
            obs, rwd, done, env_info = env.forward()
            for i_step in range(trace_horizon+1):

                # Get step's actions ----------------------

                # Record Execution. Useful for kinesthetic demonstrations on hardware
                if mode=='record':
                    act = env.action_space.sample() # dummy random sample

                # Directly create the scene
                elif mode=='render':
                    act = path_data['actions'][i_step]

                # Apply actions in open loop
                elif mode=='playback':
                    if rollout_format=='RoboSet':
                        act = np.concatenate([path_data['ctrl_arm'][i_step], path_data['ctrl_ee'][i_step]])
                    elif rollout_format=='RoboHive':
                        act = path_data['actions'][i_step]

                # Recover actions from states
                elif mode=='recover':
                    # assumes position controls
                    if rollout_format=='RoboSet':
                        act = np.concatenate([path_data['qp_arm'][i_step], path_data['qp_ee'][i_step]])
                    elif rollout_format=='RoboHive':
                        act = path_data['env_infos']['obs_dict']['qp'][i_step]
                    if noise_scale:
                        act = act + env.env.np_random.uniform(high=noise_scale, low=-noise_scale, size=len(act)).astype(act.dtype)
                    if env.normalize_act:
                        act = env.robot.normalize_actions(controls=act)

                # nan actions for last log entry
                if i_step == trace_horizon:
                    act = np.nan*np.ones(env.action_space.shape)

                # log values at time=t ----------------------------------
                if compress_paths:
                    obs = [] # don't save obs, env_infos has obs_dict
                    if 'state' in env_info.keys(): del env_info['state']  # don't save state, obs_dict has env necessities

                # log: time, obs, act, rwd, info, done
                datum_dict = dict(
                        time=env.time,
                        observations=obs,
                        actions=act.copy(),
                        rewards=rwd,
                        env_infos=env_info,
                        done=done,
                    )
                trace.append_datums(group_key=path_name,dataset_key_val=datum_dict)

                # log: offscreen frames
                if render =='offscreen':
                    curr_frame = env.sim.renderer.render_offscreen(
                        camera_id=camera_name,
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=0
                    )
                    frames[i_step,:,:,:] = curr_frame
                    # print(i_step, end=', ', flush=True)

                # step/forward env using actions from t=>t+1 ----------------------
                if i_step < trace_horizon and mode=='render':
                    env.sim.data.time = path_data['time'][i_step]
                    env.env.set_env_state(state_t[i_step])
                    obs, rwd, done, env_info = env.forward()
                    ep_rwd += rwd
                elif i_step < trace_horizon: #incase last step actions (nans) can cause issues in step
                    act = act.astype(np.float32, copy=False)
                    obs, rwd, done, env_info = env.step(act)
                    ep_rwd += rwd

            # save offscreen buffers as video
            if render == 'offscreen':
                file_name = output_dir + 'rollout' + str(path_name) + ".mp4"
                inputdict={"-r": str(1/env.dt)}
                # check if the platform is OS -- make it compatible with quicktime
                if platform == "darwin":
                    skvideo.io.vwrite(file_name, np.asarray(frames), inputdict=inputdict, outputdict={"-pix_fmt": "yuv420p"},)
                else:
                    skvideo.io.vwrite(file_name, np.asarray(frames), inputdict=inputdict)
                print("\nSaved: " + file_name)

            # Finish rollout
            print(f"Finishing {path_name} rollout in {(time.time()-ep_t0):0.4} sec. Total rewards {ep_rwd}")

        # Finish loop
        print("Finished rollout loop:{}".format(i_loop))

    # plot paths ???: Needs upgrade to the new logger
    trace.stack()
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    # plot paths
    if plot_paths:
        file_name = os.path.join(output_dir, output_name + '{}'.format(time_stamp))
        plotnsave_paths(trace.trace, env=env, fileName_prefix=file_name)

    # Close and save paths
    trace.close()
    if save_paths:
        file_name = os.path.join(output_dir, output_name + '{}_paths.h5'.format(time_stamp))
        trace.save(trace_name=file_name)


if __name__ == '__main__':
    main()
