# Instructions for the generation of DM dataset

The generation of the DM trajectory consists of 2 steps:

1. Generate Inverse Kinematics (IK) files from a source GRAB file:
``` bash
python playback_mocap.py --human_motion_file <source file> <destination> --start X --end Y --sim_name <name of the environment> --length Z --no_viewer
```
This routine also generate files of the mocap. See also sample bash routing `generate_ik_test.sh`

2. Generate MyoDM trajectories

``` bash
python generate_myodm_traj.py
```
