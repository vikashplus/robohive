# Motion files for the TCDM tasks
This folder contains the motion files containing manipulation behaviors. These files can be used to create tasks. For example, in this repo we have used it to create the TCDM benchmarks.


## Naming convention for the motion files
`<robot_name>_<object_name>_<motion_name>.npz`
- `robot_name` can be none if robot reference motion isn't included. Useful for strict task specification.
- `object_name` can be none if object reference motion isn't included. Useful for robot pose tracking.
- Both `robot_name` and `object_name` can't be none as one of them is required for task specification

## Data convention for the motion files
- Motion data is assumed to be compatible with the corresponding simulation scene. This implies that data is the same frame of reference as the world co-ordinate of the scene.

## Data format in the motion files
```python
    'time',        # float(N) ==> Time stamp for the reference
    'robot',       # shape(N, n_robot_jnt) ==> robot trajectory
    'object',      # shape(M, n_objects_jnt) ==> object trajectory
    'robot_init',  # shape(n_objects_jnt) ==> initial robot pose, if different from robot[0,n_robot_jnt]
    'object_init'  # shape(n_objects_jnt) ==> initial object, if different from object[0,n_object_jnt]
```
- Both `robot` and `object` can't be none as one of them is required for task specification
- Pose at t=0 is used if initial poses isn't explicitely specified.