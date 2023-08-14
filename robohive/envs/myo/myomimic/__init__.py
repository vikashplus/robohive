from gym.envs.registration import register
import collections
import os
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))


# Task specification format
task_spec = collections.namedtuple('task_spec',
        ['name',        # task_name
         'robot',       # robot name
         'object',      # object name
         'motion',      # motion reference file path
         ])

# MyoDM tasks
MyoLegs_task_spec = (
    task_spec(name='MyoLegJump-v0', robot='MyoLeg', object=None, motion='Subj05jumpIK.pkl'),
    task_spec(name='MyoLegLunge-v0', robot='MyoLeg', object=None, motion='Subj05lungeIK.pkl'),
    task_spec(name='MyoLegSquat-v0', robot='MyoLeg', object=None, motion='Subj05squatIK.pkl'),
    task_spec(name='MyoLegLand-v0', robot='MyoLeg', object=None, motion='Subj05landIK.pkl'),
    task_spec(name='MyoLegRun-v0', robot='MyoLeg', object=None, motion='Subj05run_99IK.pkl'),
    task_spec(name='MyoLegWalk-v0', robot='MyoLeg', object=None, motion='Subj05walk_09IK.pkl'),
)
# Register MyoHand envs using reference motion
def register_myoleg_trackref(task_name, object_name, motion_path=None):
    # Track reference motion
    # print("'"+task_name+"'", end=", ")
    register(
        id=task_name,
        entry_point='robohive.envs.myo.myomimic.myomimic_v0:TrackEnv',
        max_episode_steps=75, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': curr_dir+'/../../../simhive/myo_sim/myoleg/myoleg_v0.5(mj120).mjb',
                'reference':curr_dir+'/data/'+motion_path,
            }
    )
for task_name, robot_name, object_name, motion_path in MyoLegs_task_spec:
    register_myoleg_trackref(task_name, object_name, motion_path)
