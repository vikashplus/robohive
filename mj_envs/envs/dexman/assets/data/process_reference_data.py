"""
Script to process reference data into reference format
reference = collections.namedtuple('reference',
        ['time',        # int
         'robot',       # shape(N, n_robot_jnt) ==> robot trajectory
         'object',      # shape(M, n_objects_jnt) ==> object trajectory
         'robot_init',  # shape(n_objects_jnt) ==> initial robot pose
         'object_init'  # shape(n_objects_jnt) ==> initial object
         ])
"""
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))


sim_dt = 0.002
n_robot_jnt = 30

def load(reference_path):
    return {k:v for k, v in np.load(reference_path).items()}

def proces_n_save(data, new_path_name):
    time = np.arange(0, data['length']) * data['SIM_SUBSTEPS']*sim_dt
    data_obj = np.concatenate([data['object_translation'], data['object_orientation']], axis=1)
    np.savez(new_path_name,
             time=time,
             robot=data['s_0'][None,:n_robot_jnt],
             object=data_obj,
             robot_int=data['s_0'][:n_robot_jnt],
             object_int=data['s_0'][n_robot_jnt:],
             )

path_name = curr_dir+'/banana_pass1.npz'
new_path_name = curr_dir+'/banana_pass1_new.npz'
data = load(path_name)
proces_n_save(data, new_path_name)