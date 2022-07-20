DESC="""
Script to recover images and videos from the saved pickle files
 - python utils/render_paths.py -p  paths.pickle -f mp4 -c right
 - python utils/render_paths.py -p  paths.h5 -f rgb -c left
"""
import pickle
from PIL import Image
import os
import numpy as np
from traitlets import List
import skvideo.io
import click
import h5py

@click.command(help=DESC)
@click.option('-p', '--rollout_path', type=str, help='absolute path of the rollout (h5/pickle)', default=None)
@click.option('-fs', '--frame_size', type=str, help='image size', default="[240, 424]")
@click.option('-f', '--format', type=click.Choice(['rgb', 'mp4']), help='format to save', default="mp4")
@click.option('-c', '--cam', type=click.Choice(['left', 'right', 'center']), help='camera to render', default="left")
def main(rollout_path, frame_size, format, cam):

    output_dir = os.path.dirname(rollout_path)
    rollout_name = os.path.split(rollout_path)[-1]
    output_name, output_type = os.path.splitext(rollout_name)
    frame_size = eval(frame_size)
    cam_name = 'rgb:{}_cam:240x424:flat'.format(cam)

    if output_type=='.h5':
        paths = h5py.File(rollout_path, 'r')
    elif output_type=='.pickle':
        paths = pickle.load(open(rollout_path, 'rb'))
    else:
        raise TypeError("Unknown path format. Check file")


    # Run through all trajs in the paths
    for i_path, path in enumerate(paths):
        if output_type=='.h5':
            path = paths[path]
        if i_path == 0:
            path_horizon = path['actions'].shape[0]
            frames = np.zeros((path_horizon, frame_size[0], frame_size[1], 3), dtype=np.uint8)

        if format == "mp4":
            file_name = os.path.join(output_dir, output_name+'{}{}.mp4'.format(i_path, cam))
            print("Recovering frames:", end="")
            for t in range(path_horizon):
                frames[t,:,:,:] = path['env_infos']['obs_dict'][cam_name][t].reshape(frame_size[0], frame_size[1], 3)
                print(t, end=",")
            frames[frames==255] = 254 # remove rendering artifact due to saturation
            skvideo.io.vwrite(file_name, np.asarray(frames))
            print("\nSaving: {}".format(file_name))

        elif format == "rgb":
            print("Recovering frames:", end="")
            for t in range(path_horizon):
                file_name = os.path.join(output_dir, output_name+'{}{}{}.png'.format(i_path, cam, t))
                img =  path['env_infos']['obs_dict'][cam_name][t].reshape(frame_size[0], frame_size[1], 3)
                image = Image.fromarray(img)
                image.save(file_name)
                print(t, end=",")
            print(": Done")
        else:
            raise TypeError("Unknown format")


if __name__ == '__main__':
    main()
