DESC="""
Utility scripts for paths
"""

import numpy as np
import click
import os
import glob
import pickle
from data_tools.writer import DataWriter
import h5py

@click.command(help=DESC)
@click.option('-p', '--rollout_path', type=str, help='single path or folder with paths', required=True)
@click.option('-o', '--output_dir', type=str, default=None, help=('Directory to save the outputs'))
@click.option('-v', '--verify', type=bool, default=False, help=('Verify the saved file'))
def main(rollout_path, output_dir, verify):

    # resolve output dir
    if output_dir == None: # overide the default
        output_dir = os.path.dirname(rollout_path)

    # resolve rollout_paths
    if os.path.isfile(rollout_path):
        rollout_paths = [rollout_path]
    else:
        rollout_paths = glob.glob(os.path.join(rollout_path, '*.pickle'))

    # Parse all rollouts
    for rollout_path in rollout_paths:

        # parse all paths
        print('Parsing: ', rollout_path)
        paths = pickle.load(open(rollout_path, 'rb'))
        rollout_name = os.path.split(rollout_path)[-1]
        output_name = os.path.splitext(rollout_name)[0]
        output_path = os.path.join(output_dir, output_name + '.h5')
        print('Saving:  ', output_path)

        # start a h5 writer for this path
        writer = DataWriter(output_path)
        for i_path, path in enumerate(paths):
            path = writer.flatten_dict('', path)

            # parse all frames in this path/ tiral
            horizon = path['actions'].shape[0]
            for h in range(horizon):
                frame_dict ={}
                for key, val in path.items():
                    # print(key)
                    frame_dict[key] = [] if len(val)==0 else val[h]
                writer.add_frame(**frame_dict)
            writer.write_trial('Trial'+str(i_path))

        # close the h5 writer for this path
        del writer

        # Read back and verify a few keys
        if verify:
            with h5py.File(output_path, "r") as f:
                # List all groups
                for key, val in f.items():
                    print(key, ":: ")
                    for key_val, val_val in val.items():
                        print('\t', key_val, '\t', val_val)

    print("Finished Processing")

if __name__ == '__main__':
    main()



