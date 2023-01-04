# RoboHive Logger
RoboHive provides a custom logger to facilitates working with grouped data. Logger consists of a `name` for the logs, `group` for organizing the data, and `dataset` for storing data. Attached below is logger's general schema -
```
name: {
    group{g1: dataset{k1:v1}, dataset{k2:v2}, ...}
    group{g2: dataset{kx:vx}, dataset{ky:vy}, ...}
    ...
}
```
Lets understand RoboHive's Logger details using its most common usecase -- i.e. recording Robot trajectories.


## Robot Trajectories (Rollouts)
Robot trajectories, known as rollouts, are represented in RoboHive as time aligned Tuples. Note that a trajectory of horizon `h` we will have `h+1` tuples to account for the initial and the final states. We record the last actions as `NaN`
```
<t(0)   : s(0),     a(0)>
<t(1)   : s(1),     a(1)>
...
<t(h-1) : s(h-1),   a(h-1)>
<t(h)   : s(h),     NaN>
```

## Robot Trajectories (Rollouts) in RoboHive Logger
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


## Logger Usage:
### 1. Record rollouts
Record grouped datasets as they are generated
```
from mj_envs.logger.grouped_datasets import Trace as Trace
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
