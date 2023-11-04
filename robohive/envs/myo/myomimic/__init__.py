from gym.envs.registration import register
import collections
import os
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))

# fetch datasets
from robohive.utils.import_utils import fetch_git
motion_dir = fetch_git(repo_url="git@github.com:myolab/myomotion.git",
                       commit_hash="d29b8129f561ffe9b1e1bbef6457886af28bc800",
                       clone_directory="myomotion")

# Task specification format
task_spec = collections.namedtuple('task_spec',
        ['name',        # task_name
         'robot',       # robot name
         'object',      # object name
         'motion',      # motion reference file path
         ])


MyoLegs_task_spec = (
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04jumpIK-v0', motion='Subj04jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04landIK-v0', motion='Subj04landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04lungeIK-v0', motion='Subj04lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04run_63IK-v0', motion='Subj04run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04run_81IK-v0', motion='Subj04run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04run_99IK-v0', motion='Subj04run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04squatIK-v0', motion='Subj04squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04walk_09IK-v0', motion='Subj04walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04walk_18IK-v0', motion='Subj04walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04walk_27IK-v0', motion='Subj04walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04walk_36IK-v0', motion='Subj04walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04walk_45IK-v0', motion='Subj04walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj04walk_54IK-v0', motion='Subj04walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05jumpIK-v0', motion='Subj05jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05landIK-v0', motion='Subj05landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05lungeIK-v0', motion='Subj05lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05run_63IK-v0', motion='Subj05run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05run_81IK-v0', motion='Subj05run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05run_99IK-v0', motion='Subj05run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05squatIK-v0', motion='Subj05squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05walk_09IK-v0', motion='Subj05walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05walk_18IK-v0', motion='Subj05walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05walk_27IK-v0', motion='Subj05walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05walk_36IK-v0', motion='Subj05walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05walk_45IK-v0', motion='Subj05walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj05walk_54IK-v0', motion='Subj05walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06jumpIK-v0', motion='Subj06jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06landIK-v0', motion='Subj06landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06lungeIK-v0', motion='Subj06lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06run_63IK-v0', motion='Subj06run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06run_81IK-v0', motion='Subj06run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06run_99IK-v0', motion='Subj06run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06squatIK-v0', motion='Subj06squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06walk_09IK-v0', motion='Subj06walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06walk_18IK-v0', motion='Subj06walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06walk_27IK-v0', motion='Subj06walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06walk_36IK-v0', motion='Subj06walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06walk_45IK-v0', motion='Subj06walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj06walk_54IK-v0', motion='Subj06walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07jumpIK-v0', motion='Subj07jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07landIK-v0', motion='Subj07landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07lungeIK-v0', motion='Subj07lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07run_63IK-v0', motion='Subj07run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07run_81IK-v0', motion='Subj07run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07run_99IK-v0', motion='Subj07run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07squatIK-v0', motion='Subj07squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07walk_09IK-v0', motion='Subj07walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07walk_18IK-v0', motion='Subj07walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07walk_27IK-v0', motion='Subj07walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07walk_36IK-v0', motion='Subj07walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07walk_45IK-v0', motion='Subj07walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj07walk_54IK-v0', motion='Subj07walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08jumpIK-v0', motion='Subj08jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08landIK-v0', motion='Subj08landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08lungeIK-v0', motion='Subj08lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08run_63IK-v0', motion='Subj08run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08run_81IK-v0', motion='Subj08run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08run_99IK-v0', motion='Subj08run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08squatIK-v0', motion='Subj08squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08walk_09IK-v0', motion='Subj08walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08walk_18IK-v0', motion='Subj08walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08walk_27IK-v0', motion='Subj08walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08walk_36IK-v0', motion='Subj08walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08walk_45IK-v0', motion='Subj08walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj08walk_54IK-v0', motion='Subj08walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09jumpIK-v0', motion='Subj09jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09landIK-v0', motion='Subj09landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09lungeIK-v0', motion='Subj09lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09run_63IK-v0', motion='Subj09run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09run_81IK-v0', motion='Subj09run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09run_99IK-v0', motion='Subj09run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09squatIK-v0', motion='Subj09squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09walk_09IK-v0', motion='Subj09walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09walk_18IK-v0', motion='Subj09walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09walk_27IK-v0', motion='Subj09walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09walk_36IK-v0', motion='Subj09walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09walk_45IK-v0', motion='Subj09walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj09walk_54IK-v0', motion='Subj09walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10jumpIK-v0', motion='Subj10jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10landIK-v0', motion='Subj10landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10lungeIK-v0', motion='Subj10lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10run_63IK-v0', motion='Subj10run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10run_81IK-v0', motion='Subj10run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10run_99IK-v0', motion='Subj10run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10squatIK-v0', motion='Subj10squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10walk_09IK-v0', motion='Subj10walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10walk_18IK-v0', motion='Subj10walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10walk_27IK-v0', motion='Subj10walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10walk_36IK-v0', motion='Subj10walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10walk_45IK-v0', motion='Subj10walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj10walk_54IK-v0', motion='Subj10walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11jumpIK-v0', motion='Subj11jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11landIK-v0', motion='Subj11landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11lungeIK-v0', motion='Subj11lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11run_63IK-v0', motion='Subj11run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11run_81IK-v0', motion='Subj11run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11run_99IK-v0', motion='Subj11run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11squatIK-v0', motion='Subj11squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11walk_09IK-v0', motion='Subj11walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11walk_18IK-v0', motion='Subj11walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11walk_27IK-v0', motion='Subj11walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11walk_36IK-v0', motion='Subj11walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11walk_45IK-v0', motion='Subj11walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj11walk_54IK-v0', motion='Subj11walk_54IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12jumpIK-v0', motion='Subj12jumpIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12landIK-v0', motion='Subj12landIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12lungeIK-v0', motion='Subj12lungeIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12run_63IK-v0', motion='Subj12run_63IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12run_81IK-v0', motion='Subj12run_81IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12run_99IK-v0', motion='Subj12run_99IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12squatIK-v0', motion='Subj12squatIK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12walk_09IK-v0', motion='Subj12walk_09IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12walk_18IK-v0', motion='Subj12walk_18IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12walk_27IK-v0', motion='Subj12walk_27IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12walk_36IK-v0', motion='Subj12walk_36IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12walk_45IK-v0', motion='Subj12walk_45IK.npz'),
    task_spec(robot='myolegs_abd_v0.56(mj236).mjb', object=None, name='MyoLegsSubj12walk_54IK-v0', motion='Subj12walk_54IK.npz')
)

# Register MyoLegs envs using reference motion
def register_myoleg_trackref(task_name, robot_name, object_name, motion_path):
    register(
        id=task_name,
        entry_point='robohive.envs.myo.myomimic.myomimic_v0:TrackEnv',
        max_episode_steps=75, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': os.path.join(motion_dir, 'myolegs_abd', robot_name),
                'reference':  os.path.join(motion_dir, 'myolegs_abd', motion_path),
            }
    )
for task_name, robot_name, object_name, motion_path in MyoLegs_task_spec:
    register_myoleg_trackref(task_name, robot_name, object_name, motion_path)
