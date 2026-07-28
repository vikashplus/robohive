""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import glob
import json
import os
import pickle

import click
import h5py
import numpy as np
import skvideo.io
from PIL import Image

from robohive.utils import gym
from robohive.utils.dict_utils import dict_numpify, flatten_dict
from robohive.logger.grouped_datasets import Trace

#TODO: Harmonize names, remove rollout_paths, use path for one and paths for multiple

ROLLOUT_EXTENSIONS = ('.h5', '.pickle')


# Normalize a loaded paths object (list or dict-like) into a name->path dict
def _normalize_paths(paths):
    if isinstance(paths, (list, tuple)):
        return {'Trial{}'.format(i): path for i, path in enumerate(paths)}
    if isinstance(paths, Trace):
        return dict(paths.trace)
    return paths


# Load a single rollout file (.h5 / .pickle) into a normalized path object
def load_rollout_file(rollout_path:str):
    """
    Load a single rollout file (.h5 or .pickle) from disk.

    Args:
        rollout_path (str): absolute path to a rollout file
    Returns:
        dict-like object mapping path/trial name to its data
    """
    ext = os.path.splitext(rollout_path)[-1]
    if ext == '.h5':
        paths = h5py.File(rollout_path, 'r')
    elif ext == '.pickle':
        paths = pickle.load(open(rollout_path, 'rb'))
    else:
        raise TypeError("Unknown rollout format:{}. Supported formats: {}".format(ext, ROLLOUT_EXTENSIONS))
    return _normalize_paths(paths)


# Harmonized preprocessor: resolve a path_handle into a loaded path object
def resolve_path_handle(path_handle, extensions=ROLLOUT_EXTENSIONS, return_sources=False):
    """
    Resolve a path_handle into a normalized, loaded path object.

    path_handle can be-
        - an already loaded path object (dict / list / h5py.File / Trace.trace, ...)
        - a str path to a single rollout file (.h5 or .pickle)
        - a str path to a directory containing rollout files (.h5 / .pickle),
          searched recursively through subdirectories, and all matching files
          are loaded and merged

    Args:
        path_handle: handle to resolve (see above)
        extensions: rollout file extensions to scan for when path_handle is a directory
        return_sources (bool): if True, also return a dict mapping each path/trial name to a
            (source_dir, source_file_stem) tuple describing the file it was loaded from
            ((None, None) if path_handle was already a loaded object with no known file location)
    Returns:
        dict-like object mapping path/trial name to its data
        (dict mapping path/trial name to its (source_dir, source_file_stem), if return_sources)
    """
    if isinstance(path_handle, str):
        if os.path.isdir(path_handle):
            rollout_files = sorted(
                f for ext in extensions
                for f in glob.glob(os.path.join(path_handle, '**', '*'+ext), recursive=True))
            assert len(rollout_files) > 0, \
                "No rollout files (formats:{}) found in directory:{} (searched recursively)".format(extensions, path_handle)
            paths, sources = {}, {}
            for rollout_file in rollout_files:
                prefix = os.path.splitext(os.path.relpath(rollout_file, path_handle))[0]
                file_dir = os.path.dirname(rollout_file)
                file_stem = os.path.splitext(os.path.basename(rollout_file))[0]
                for name, data in load_rollout_file(rollout_file).items():
                    key = '{}/{}'.format(prefix, name)
                    paths[key] = data
                    sources[key] = (file_dir, file_stem)
            return (paths, sources) if return_sources else paths
        elif os.path.isfile(path_handle):
            paths = load_rollout_file(path_handle)
            file_dir = os.path.dirname(os.path.abspath(path_handle))
            file_stem = os.path.splitext(os.path.basename(path_handle))[0]
            return (paths, {key: (file_dir, file_stem) for key in paths.keys()}) if return_sources else paths
        else:
            raise FileNotFoundError("Path not found:{}".format(path_handle))
    else:
        # already a loaded path object; source location is unknown
        paths = _normalize_paths(path_handle)
        return (paths, {key: None for key in paths.keys()}) if return_sources else paths


# Harmonized preprocessor: resolve an env_handle into a loaded, unwrapped env
def resolve_env_handle(env_handle, env_args=None):
    """
    Resolve an env_handle into a loaded, unwrapped env object.

    env_handle can be-
        - an already loaded env object (wrapped or unwrapped)
        - a str env_name to be created via gym.make

    Args:
        env_handle: handle to resolve (see above)
        env_args (str, optional): kwargs (as a str-eval'able dict) to pass to gym.make, when env_handle is a str
    Returns:
        unwrapped env object, or None if env_handle is None
    """
    if env_handle is None:
        return None
    elif isinstance(env_handle, str):
        env = gym.make(env_handle) if env_args is None else gym.make(env_handle, **(eval(env_args)))
        return env.unwrapped
    else:
        # already a loaded env object
        return env_handle.unwrapped if hasattr(env_handle, 'unwrapped') else env_handle


# Check the horizon for teleOp / Hardware experiments
def plot_horizon(path_handle, env_handle, output_dir=None, output_name='', env_args=None):
    """
    Check the horizon for teleOp / Hardware experiments

    Args:
        path_handle: loaded path object, or path to a rollout file/ directory of rollout files
        env_handle: loaded (unwrapped) env object, or an env_name to be created via gym.make
        output_dir (str, optional): Directory to save the outputs. Defaults to path_handle's
            directory (or cwd if path_handle is not a str path).
        output_name (str, optional): prefix to use in the output filename
        env_args (str, optional): kwargs (as a str-eval'able dict) to pass to gym.make, when env_handle is a str
    Saves:
        output_dir/output_name + '_horizon.pdf'
    """
    paths = list(resolve_path_handle(path_handle).values())
    env = resolve_env_handle(env_handle, env_args=env_args)

    if output_dir is None:
        if isinstance(path_handle, str):
            output_dir = path_handle if os.path.isdir(path_handle) else (os.path.dirname(os.path.abspath(path_handle)))
        else:
            output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    fileName_prefix = os.path.join(output_dir, output_name or '')

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 5})

    if "time" in paths[0]['env_infos']:
        horizon = np.zeros(len(paths))

        # plot timesteps
        plt.clf()

        rl_dt_ideal = env.dt
        for i, path in enumerate(paths):
            dt = path['env_infos']['time'][1:] - path['env_infos']['time'][:-1]
            horizon[i] = path['env_infos']['time'][-1] - path['env_infos'][
                'time'][0]
            plt.plot(
                path['env_infos']['time'][1:],
                dt,
                '-',
                alpha=.8,
                label=('hor=%1.2f' % horizon[i]))
        plt.plot(
            np.array([0, max(horizon)]),
            rl_dt_ideal * np.ones(2),
            'g', alpha=.5,
            linewidth=2.0,
            label='ideal dt')

        plt.legend(loc='upper right')
        plt.ylabel('time step (sec)')
        plt.xlabel('time (sec)')
        plt.ylim(rl_dt_ideal - 2*rl_dt_ideal, rl_dt_ideal + 3*rl_dt_ideal)
        plt.suptitle('Timestep profile for %d rollouts' % len(paths))

        file_name = fileName_prefix + '_timesteps.pdf'
        plt.savefig(file_name)
        print("Saved:", file_name)

        # plot horizon
        plt.clf()
        plt.plot(
            np.array([0, len(paths)]),
            env.horizon * rl_dt_ideal * np.ones(2),
            'g',
            linewidth=5.0,
            label='ideal')
        plt.bar(np.arange(0, len(paths)), horizon, label='observed')
        plt.ylabel('rollout duration (sec)')
        plt.xlabel('rollout id')
        plt.legend()
        plt.suptitle('Horizon distribution for %d rollouts' % len(paths))

        file_name = fileName_prefix + '_horizon.pdf'
        plt.savefig(file_name)
        print("Saved:", file_name)


# 2D-plot of paths detailing obs, act, rwds across time
def plot(path_handle, env_handle=None, output_dir=None, output_name='', env_args=None):
    """
    2D-plot of paths detailing obs, act, rwds across time

    Args:
        path_handle: loaded path object, or path to a rollout file/ directory of rollout files
        env_handle: loaded (unwrapped) env object, or an env_name to be created via gym.make
        output_dir (str, optional): Directory to save the outputs. If not provided, each
            trial's plot is saved next to the rollout file it came from. If provided, the
            source directory structure (for directory path_handles) is mirrored under it.
        output_name (str, optional): prefix to use in the output filenames. If not provided,
            defaults to the name of the rollout file the trial came from (when known).
        env_args (str, optional): kwargs (as a str-eval'able dict) to pass to gym.make, when env_handle is a str

    Saves:
        one pdf per trial, named <output_name or source file name>_<trial_name> + '.pdf'
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 5})

    paths, sources = resolve_path_handle(path_handle, return_sources=True)
    env = resolve_env_handle(env_handle, env_args=env_args)
    is_dir_handle = isinstance(path_handle, str) and os.path.isdir(path_handle)
    for path_name, path in paths.items():
        leaf_name = path_name.split('/')[-1]
        source_dir, source_stem = sources.get(path_name) or (None, None)
        if output_dir is not None:
            if is_dir_handle and source_dir is not None:
                rel_dir = os.path.relpath(source_dir, path_handle)
                save_dir = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
            else:
                save_dir = output_dir
        else:
            save_dir = source_dir or '.'
        os.makedirs(save_dir, exist_ok=True)
        name_prefix = output_name or source_stem or ''
        file_stem = '_'.join(p for p in (name_prefix, leaf_name) if p)
        plt.clf()

        # observations
        obs_keys = sorted(
            key for key in path['env_infos']['obs_dict'].keys()
            if path['env_infos']['obs_dict'][key].ndim < 3)
        nplt1 = len(obs_keys)
        time = np.asarray(path['env_infos']['time'])
        print(obs_keys)
        for iplt1, key in enumerate(obs_keys):
            ax = plt.subplot(nplt1, 2, iplt1 * 2 + 1)
            if iplt1 == 0:
                plt.ylabel('Observations')
            ax.yaxis.tick_right()
            if path['env_infos']['obs_dict'][key].size > 0:
                plt.plot(time, np.asarray(path['env_infos']['obs_dict'][key]), label=key)
            ax.set_xlim(time[0], time[-1])
            if iplt1 != (nplt1 - 1):
                ax.axes.xaxis.set_ticklabels([])
            if iplt1 == 0:
                plt.title('Observations')
            ax.yaxis.tick_right()
            if path['env_infos']['obs_dict'][key].ndim<3:
                plt.plot(
                    path['env_infos']['time'],
                    path['env_infos']['obs_dict'][key],
                    label=key)
            # plt.ylabel(key)
            plt.text(0.01, .01, f"{key}{path['env_infos']['obs_dict'][key].shape}", transform=ax.transAxes)
        plt.xlabel('time (sec)')

        # actions
        nplt2 = 3
        ax = plt.subplot(nplt2, 2, 2)
        ax.set_prop_cycle(None)
        # h4 = plt.plot(path['env_infos']['time'], env.act_mid + path['actions']*env.act_rng, '-', label='act') # plot scaled actions
        h4 = plt.plot(
            time, np.asarray(path['actions']), '-',
            label='act')  # plot normalized actions
        plt.ylabel('actions')
        ax.axes.xaxis.set_ticklabels([])
        ax.yaxis.tick_right()

        # rewards/ scores
        if "score" in path['env_infos']:
            ax = plt.subplot(nplt2, 2, 6)
            plt.plot(
                time,
                np.asarray(path['env_infos']['score']),
                label='score')
            plt.xlabel('time')
            plt.ylabel('score')
            ax.yaxis.tick_right()

        if "rwd_dict" in path['env_infos']:
            ax = plt.subplot(nplt2, 2, 4)
            ax.set_prop_cycle(None)
            for key in sorted(path['env_infos']['rwd_dict'].keys()):
                plt.plot(
                    time,
                    np.asarray(path['env_infos']['rwd_dict'][key]),
                    label=key)
            plt.legend(
                loc='upper left',
                fontsize='x-small',
                bbox_to_anchor=(.75, 0.25),
                borderaxespad=0.)
            ax.axes.xaxis.set_ticklabels([])
            plt.ylabel('rewards')
            ax.yaxis.tick_right()

        if env and hasattr(env, "rwd_keys_wt"):
            ax = plt.subplot(nplt2, 2, 6)
            ax.set_prop_cycle(None)
            for key in sorted(env.rwd_keys_wt.keys()):
                plt.plot(
                    time,
                    np.asarray(path['env_infos']['rwd_dict'][key])*env.rwd_keys_wt[key],
                    label=key)
            plt.legend(
                loc='upper left',
                fontsize='x-small',
                bbox_to_anchor=(.75, 0.25),
                borderaxespad=0.)
            ax.axes.xaxis.set_ticklabels([])
            plt.ylabel('wt*rewards')
            ax.yaxis.tick_right()

        file_name = os.path.join(save_dir, file_stem + '.pdf')
        plt.savefig(file_name)
        print("saved ", file_name)


# Render frames/videos
def render(path_handle, render_format:str="mp4", cam_names:list=["left"], output_dir=None, output_name=None):
    """
    Render the frames from a given rollout.

    Parameters:
        path_handle: loaded path object, or path to a rollout file/ directory of rollout files (h5/pickle).
        render_format (str, optional): Format to save the rendered frames. Default is "mp4".
        cam_names (list, optional): List of cameras to render. Default is ["left"]. Example ['left', 'right', 'top', 'Franka_wrist']
        output_dir (str, optional): Directory to save the outputs. Defaults to the rollout's directory (or cwd if path_handle is not a file path).
        output_name (str, optional): Prefix to use for the output filenames. Defaults to the rollout's name (or "rollout" if path_handle is not a file path).

    Returns:
        None

    Raises:
        TypeError: If the path format is unknown.

    Notes:
        - The frames are saved in the specified render format.
        - The rendered frames can be saved as an mp4 video or as individual RGB images.
        - The frames are rendered for each camera specified in the cam_names list.
        - The output file names are generated based on the rollout name and the camera names.

    Example:
        render(path_handle="/path/to/rollout.h5", render_format="mp4", cam_names=["left", "right"])
    """

    if isinstance(path_handle, str) and os.path.isfile(path_handle):
        output_dir = output_dir or os.path.dirname(path_handle) or '.'
        output_name = output_name or os.path.splitext(os.path.basename(path_handle))[0]
    else:
        output_dir = output_dir or '.'
        output_name = output_name or 'rollout'
    file_name = os.path.join(output_dir, output_name+"_"+"-".join(cam_names))

    paths = resolve_path_handle(path_handle)

    # Run through all trajs in the paths
    for i_path, (path_name, path) in enumerate(paths.items()):

        if 'data' in path.keys():
            data = path['data']
            path_horizon = data['time'].shape[0]
        else:
            obs_dict = path['env_infos']['obs_dict']
            data = {key: obs_dict[key] for key in obs_dict.keys()}
            if 'visual_dict' in path['env_infos'].keys():
                visual_dict = path['env_infos']['visual_dict']
                data.update({key: visual_dict[key] for key in visual_dict.keys()})
            path_horizon = path['env_infos']['time'].shape[0]

        # find full key name
        data_keys = data.keys()
        cam_keys = []
        for cam_name in cam_names:
            cam_key = None
            for key in data_keys:
                if cam_name in key and 'rgb' in key:
                   cam_key = key
                   break
            assert cam_key != None, "Cam: {} not found in data. Available keys: [{}]".format(cam_name, data_keys)
            cam_keys.append(key)


        # pre allocate buffer
        if i_path==0:
            height, width, _ = data[cam_keys[0]][0].shape
            frame_tile = np.zeros((height, width*len(cam_keys), 3), dtype=np.uint8)
            if render_format == "mp4":
                frames = np.zeros((path_horizon, height, width*len(cam_keys), 3), dtype=np.uint8)

        # Render
        print("Recovering {} frames:".format(render_format), end="")
        for t in range(path_horizon):
            # render single frame
            for i_cam, cam_key in enumerate(cam_keys):
                frame_tile[:,i_cam*width:(i_cam+1)*width, :] = data[cam_key][t]
            # process single frame
            if render_format == "mp4":
                frames[t,:,:,:] = frame_tile
            elif render_format == "rgb":
                image = Image.fromarray(frame_tile)
                image.save(file_name+"_{}-{}.png".format(i_path, t))
            else:
                raise TypeError("Unknown format")
            print(t, end=",", flush=True)

        # Save video
        if render_format == "mp4":
            file_name_mp4 = file_name+"_{}.mp4".format(i_path)
            skvideo.io.vwrite(file_name_mp4, np.asarray(frames))
            print("\nSaving: " + file_name_mp4)


# parse path from robohive format into robopen dataset format
def path2dataset(path:dict, config_path=None)->dict:
    """
    Convert Robohive format into roboset format
    """

    obs_keys = path['env_infos']['obs_dict'].keys()
    dataset = {}
    # Data =====
    dataset['data/time'] = path['env_infos']['obs_dict']['time']

    # actions
    if 'actions' in path.keys():
        dataset['data/ctrl_arm'] = path['actions'][:,:7]
        dataset['data/ctrl_ee'] = path['actions'][:,7:]

    # states
    for key in ['qp_arm', 'qv_arm', 'tau_arm', 'qp_ee', 'qv_ee']:
        if key in obs_keys:
            dataset['data/'+key] = path['env_infos']['obs_dict'][key]

    # cams
    for cam in ['left', 'right', 'top', 'wrist']:
        for key in obs_keys:
            if cam in key:
                if 'rgb:' in key:
                    dataset['data/rgb_'+cam] = path['env_infos']['obs_dict'][key]
                elif 'd:' in key:
                    dataset['data/d_'+cam] = path['env_infos']['obs_dict'][key]
    # user
    if 'user' in obs_keys:
        dataset['data/user'] = path['env_infos']['obs_dict']['user']

    # Derived =====
    pose_ee = []
    if 'pos_ee' in obs_keys or 'rot_ee' in obs_keys:
        assert ('pos_ee' in obs_keys and 'rot_ee' in obs_keys), "Both pose_ee and rot_ee are required"
        dataset['derived/pose_ee'] = np.hstack([path['env_infos']['obs_dict']['pos_ee'], path['env_infos']['obs_dict']['rot_ee']])

    # Config =====
    if config_path:
        config = json.load(open(config_path, 'rb'))
        dataset['config'] = config

    if 'user_cmt' in path.keys():
        dataset['config/solved'] = float(path['user_cmt'])

    return dataset


# Print h5 schema
def print_h5_schema(obj, name="/"):
    "Recursively find all keys in an h5py.Group or a dict-like path object."
    keys = (getattr(obj, 'name', name),)
    if isinstance(obj, h5py.Group) or isinstance(obj, dict):
        for key, value in obj.items():
            child_name = getattr(value, 'name', '{}/{}'.format(name, key))
            if isinstance(value, (h5py.Group, dict)):
                keys = keys + print_h5_schema(value, name=child_name)
            else:
                print("\t", "{0:35}".format(child_name), value)
                keys = keys + (child_name,)
    return keys


# convert paths from pickle to h5 format
def pickle2h5(rollout_path, output_dir=None, verify_output=False, h5_format:str='robohive', compress_path=False, config_path=None, max_paths=1e6):
    # rollout_path:     Single path or folder with paths
    # output_dir:       Directory to save the outputs. use path location if none.
    # verify_output:    Verify the saved file
    # h5_format:        robohive path / roboset h5s
    # compress_path:    produce smaller outputs by removing duplicate data
    # config_path:      add extra configs

   # resolve output dirzz
    if output_dir == None: # overide the default
        output_dir = os.path.dirname(rollout_path)

    # resolve rollout_paths
    if os.path.isfile(rollout_path):
        rollout_paths = [rollout_path]
    else:
        rollout_paths = glob.glob(os.path.join(rollout_path, '*.pickle'))

    # Parse all rollouts
    n_rollouts = 0
    for rollout_path in rollout_paths:

        # parse all paths
        print('Parsing: ', rollout_path)
        if n_rollouts>=max_paths:
            break

        paths = pickle.load(open(rollout_path, 'rb'))
        rollout_name = os.path.split(rollout_path)[-1]
        output_name = os.path.splitext(rollout_name)[0]
        output_path = os.path.join(output_dir, output_name + '.h5')

        paths_h5 = h5py.File(output_path, "w")

        # Robohive path format
        if h5_format == "robohive":
            for i_path, path in enumerate(paths):
                print("parsing rollout", i_path)
                trial = paths_h5.create_group('Trial'+str(i_path))
                # remove duplicate infos
                if compress_path:
                    if 'observations' in path.keys():
                        del path['observations']
                    if 'state' in path['env_infos'].keys():
                        del path['env_infos']['state']
                # flatten dict and fix resolutions
                path = flatten_dict(data=path)
                path = dict_numpify(path, u_res=None, i_res=np.int8, f_res=np.float16)
                # add trail
                for k, v in path.items():
                    trial.create_dataset(k, data=v, compression='gzip', compression_opts=4)

                n_rollouts+=1
                if n_rollouts>=max_paths:
                    break

        # RoboPen dataset format
        elif h5_format == 'roboset':
            for i_path, path in enumerate(paths):
                print("parsing rollout", i_path)
                trial = paths_h5.create_group('Trial'+str(i_path))
                dataset = path2dataset(path, config_path) # convert to robopen dataset format
                dataset = flatten_dict(data=dataset)
                dataset = dict_numpify(dataset, u_res=None, i_res=np.int8, f_res=np.float16) # numpify + data resolutions
                for k, v in dataset.items():
                    trial.create_dataset(k, data=v, compression='gzip', compression_opts=4)

                n_rollouts+=1
                if n_rollouts>=max_paths:
                    break

        else:
            raise TypeError('Unsupported h5_format')

        # close the h5 writer for this path
        print('Saving:  ', output_path)

        # Read back and verify a few keys
        if verify_output:
            with h5py.File(output_path, "r") as h5file:
                print("Printing schema read from output: ", output_path)
                keys = print_h5_schema(h5file)

    print("Finished Processing")


DESC="""
Script to recover images and videos from the saved pickle files
 - python utils/paths_utils.py -u render -p paths.pickle -rf mp4 -cn right
 - python utils/paths_utils.py -u pickle2h5 -p paths.pickle -vo True -cp True -hf robohive
 """
@click.command(help=DESC)
@click.option('-u', '--util', type=click.Choice(['plot_horizon', 'plot', 'render', 'pickle2h5', 'h5schema']), help='pick utility', required=True)
@click.option('-p', '--path', type=click.Path(exists=True), help='path_handle: absolute path of a rollout file (h5/pickle), or a directory containing rollout files', default=None)
@click.option('-e', '--env', type=str, help='env_handle: Env name to be created via gym.make (used by plot/plot_horizon)', default=None)
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-on', '--output_name', type=str, default=None, help=('Output name'))
@click.option('-od', '--output_dir', type=str, default=None, help=('Directory to save the outputs'))
@click.option('-vo', '--verify_output', type=bool, default=False, help=('Verify the saved file'))
@click.option('-hf', '--h5_format', type=click.Choice(['robohive', 'roboset']), help='format to save', default='roboset')
@click.option('-cp', '--compress_path', help='compress paths. Remove obs and env_info/state keys', default=False)
@click.option('-rf', '--render_format', type=click.Choice(['rgb', 'mp4']), help='format to save', default="mp4")
@click.option('-cn', '--cam_names', multiple=True, help='camera to render. Eg: left, right, top, Franka_wrist', default=["left", "top", "right", "wrist"])
@click.option('-ac', '--add_config', help='Add extra infos to config using as json', default=None)
@click.option('-mp', '--max_paths', type=int, help='maximum number of paths to process', default=1e6)
def util_path_cli(util, path, env, env_args, output_name, output_dir, verify_output, render_format, cam_names, h5_format, compress_path, add_config, max_paths):

    if util=='plot_horizon':
        plot_horizon(path, env, output_dir=output_dir, output_name=output_name or '', env_args=env_args)
    elif util=='plot':
        plot(path, env, output_dir=output_dir, output_name=output_name or '', env_args=env_args)
    elif util=='render':
        render(path_handle=path, render_format=render_format, cam_names=cam_names, output_dir=output_dir, output_name=output_name)
    elif util=='pickle2h5':
        pickle2h5(rollout_path=path, output_dir=output_dir, verify_output=verify_output, h5_format=h5_format, compress_path=compress_path, config_path=add_config, max_paths=max_paths)
    elif util=='h5schema':
        print("Printing schema for: ", path)
        keys = print_h5_schema(resolve_path_handle(path))
    else:
        raise TypeError("Unknown utility requested")


if __name__ == '__main__':
    util_path_cli()