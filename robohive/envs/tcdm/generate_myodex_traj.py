import numpy as np
import os
import yaml
import glob
# import matplotlib.pyplot as plt

path_myo_traj = "../myo/myodm/data/"
os.makedirs('./'+path_myo_traj+'/', exist_ok=True)

with open('./myodm_selection.yaml', 'r') as g:
    _DEXMAN_TRAJS = yaml.safe_load(g)['grab_myo']

targ_names=[]
for target_fname in _DEXMAN_TRAJS:

    if os.path.exists('./data_myoHand/ik/'+target_fname):
        file = {k: v for k, v in np.load('./data_myoHand/ik/'+target_fname).items()}

        target = target_fname.split('/')[-1][:-4]
        obj_name = target.split('_')[0]

        task_name  = target.split('_')
        task_name.remove('myo')
        task_name = ''.join(task_name[1:])

        sim_dt = 0.002; n_robot_jnt = 29
        robot = file['s'][:,:n_robot_jnt].copy()

        # Object details
        data_obj = np.concatenate([file['object_translation'], file['object_orientation']], axis=1)
        obj = file['s'][:,n_robot_jnt:]

        # time details
        horizon = robot.shape[0]

        time = np.arange(0, horizon) * file['DATA_SUBSTEPS']*sim_dt

        print("./"+path_myo_traj+"/MyoHand_"+obj_name+"_"+task_name+".npz")
        np.savez("./"+path_myo_traj+"/MyoHand_"+obj_name+"_"+task_name+".npz",
                time=time,
                robot=robot,
                object=data_obj,
                # robot_int=robot[0],
                # object_int=obj[0],
                robot_int=robot[file['grasp_frame']-1],
                object_int=obj[file['grasp_frame']-1],
                )
