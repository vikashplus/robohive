# RoboHive Logger
RoboHive provides a custom logger to facilitates working with grouped data. Robohive's logger is quite general and can also be used outside RoboHive for recording general grouped datasets (Example: RoboSet). Logger consists of a `name` for the logs, `group` for organizing the data, and `dataset` for storing data. Attached below is logger's general schema -
```
name: {
    group{g1: dataset{k1:v1}, dataset{k2:v2}, ...}
    group{g2: dataset{kx:vx}, dataset{ky:vy}, ...}
    ...
}
```
Nested groups can be created simply by using nested `group` keys. For example keys `g1/sg1`, `g1/sg2` will lead to the following nesting -
```
name: {
    group{g1:
        group{sg1: dataset{k1:v1}, dataset{k2:v2}, ...}
        group{sg2: dataset{kx:vx}, dataset{ky:vy}, ...}
    }
    ...
}
```
Note: Logger preserves the nested keys allowing data access using `data["g1/sg1"]`. The saved H5 dataset is more flexible. Data can be accessed using either `data["g1/sg1"]` or `data["g1"]["sg1"]`.

Next, let's understand RoboHive's Logger details using its most common usecase -- i.e. recording Robot trajectories.


## Robot Trajectories (Rollouts)
Robot trajectories, known as *rollouts*, are represented in RoboHive as time aligned Tuples. Note that a trajectory of horizon `h` we will have `h+1` tuples to account for the initial and the final states. We record the last actions as `NaN`.
```
<t(0)   : s(0),   a(0),   info(0)>
<t(1)   : s(1),   a(1),   info(1)>
...
<t(h-1) : s(h-1), a(h-1), info(h-1)>
<t(h)   : s(h),   NaN,    info(h)>
```

While recording Robot Trajectories using RoboHive's logger, each trajectory will form a group (Rollout0,Rollout1,...), and each variable being logged will make a dataset(sensor, action,...)
```
Rollouts_name: {
    group{Rollout0: dataset{sensor:v1}, dataset{action:v2}, ...}
    group{Rollout1: dataset{sensor:vx}, dataset{action:vy}, ...}
    ...
}
```
Note that this is a special case of RoboHive logger as:
1. Dataset names (sensor, action) are same across Rollout. RoboHive logger can have different names accross different groups in its most generic case.
2. All datasets will be of similar sizees as robot trajectories are time-aligned. RoboHive logger can have different dataset lengths within and accross different groups in its most generic case.


## Log schemas -  (1) RoboHive, (2) RoboSet
RoboHive logger is flexibile to support any schema. There are two specific schema that we follow in our ecosystem.
1. **RoboHive Schema (hdf5)**: Follow this schema if your dataset is environment specific. Explicit env specific keys `observations`, `actions`, `rewards`, `done`, `env_infos` are prominent in the schema
```
Trial1: {
    time: shape(h,), type(float)
    observations: shape(h, *), type(float)
    actions: shape(h, *), type(float)
    rewards: shape(h,), type(float)
    done: shape(h,), type(bool)

    env_infos/time: shape(h,), type(float)
    env_infos/rwd_dense: shape(h,), type(float)
    env_infos/rwd_sparse: shape(h,), type(float)
    env_infos/solved: shape(h,), type(bool)
    env_infos/done: shape(h,), type(bool)

    env_infos/obs_dict/time: shape(h,), type(float)
    env_infos/obs_dict/key*: shape(h, *), type(float)
    env_infos/proprio_dict/keys*: shape(h, *), type(float)
    env_infos/rwd_dict/key*: shape(h,*), type(float)
    env_infos/state/key*: shape(h, *), type(float)
}
```

2. **RoboSet Schema (hdf5)**: Follow this schema if your dataset is agnotic of environment/task defintion. Ideal for hardware datasets.
```
data: {
    time: shape(h,), type(int)
    ctrl_arm: shape(h, *), type(float)
    qp_arm: shape(h, *), type(float)
    qv_arm: shape(h, *), type(float)
    tau_arm: shape(h, *), type(float)

    ctrl_ee: shape(h, *), type(float)
    qp_ee: shape(h, *), type(float)
    qv_ee: shape(h, *), type(float)
    tau_ee: shape(h, *), type(float)

    rgb_left: shape(h, H, W, 3), type(uint8)
    rgb_right: shape(h, H, W, 3), type(uint8)
    rgb_top: shape(h, H, W, 3), type(uint8)
    rgb_wrist: shape(h, H, W, 3), type(uint8)

    d_left: shape(h, H, W, 3), type(uint8)
    d_right: shape(h, H, W, 3), type(uint8)
    d_top: shape(h, H, W, 3), type(uint8)
    d_wrist: shape(h, H, W, 3), type(uint8)

    user_input: shape(h, *), type(str)
}
derived:{ # save entities that can be derived/recomputed from data
    pose_ee: shape(h, *) type(float)
}
config:{ # add configurations/details to the logs
    version: shape(*) type(str)
    commit_sha: shape(*) type(str)
    task/exp name: shape(*) type(str)
    cam_config: type(dict)
}
```

3. ~~(depricated)Paths Schema (pickle): Old RoboHive schema. This schema was used in projects before RoboHive-v0.3. This schema was very close to and is being replaced by the new RoboHive schema.~~

## Logger Usage:
### 1. Record rollouts
Record grouped datasets as they are generated
```
from robohive.logger.grouped_datasets import Trace as Trace
import numpy as np

# create logger
trace = Trace("TeleOp Trajectories")

# start a new Rollout
for i_rollout in range(num_rollouts):

    # create group for the Rollout
    group_key='Rollout'+str(i_rollout);
    trace.create_group(group_key)

    # record Rollout
    for t in range(10):

        # add values to the dataset
        datum_dict = dict(
                    time=0.001*t,
                    observations=np.rand.uniform(10),
                    actions=np.rand.uniform(5),
                    rewards=np.rand.uniform(1),
                )
        trace.append_datums(group_key=group_key,dataset_key_val=datum_dict)

    # Verify length of all datasets and save logs
    trace.save("demo_rollout.h5", verify_length=True)
```
### 2. Playback Rollout
Playback the rollout again in openloop on the environment.

### 3. Render logs
Render the rollout to visualize the outcome.
