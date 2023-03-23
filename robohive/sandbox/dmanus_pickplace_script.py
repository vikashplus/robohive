# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Calculate min jerk trajectory using IK \n
    - NOTE: written for franka_busbin_v0.xml model and might not be too generic
EXAMPLE:\n
    - python tutorials/get_ik_minjerk_trajectory.py --sim_path envs/arms/franka/assets/franka_busbin_v0.xml\n
"""

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.utils.min_jerk import *
from robohive.utils.quat_math import euler2quat, euler2mat
from robohive.utils import tensor_utils
from robohive.utils.xml_utils import reassign_parent
import click
import numpy as np
import pickle
import os

# BIN_POS = np.array([-.235, 0.5, .85])
BIN_POS = np.array([0, 0.5, 1.20])
BIN_DIM = np.array([.2, .3, 0])
ARM_nJnt = 7+10
drop_z = 0.400
HAND_OPEN = np.array([4.61038e-05, -0.0378767, -0.0156091, -0.00280373, -5.5025e-08, -0.0156123, -0.00278982, -5.5025e-08, -0.0156123, -0.00278982])
HAND_CLOSE = np.array([0, 1.5199, 1.3678, 0.284, 0, 1.4029, 1.516, 0, 1.4848, 1.615])


@click.command(help=DESC)
@click.option('-s', '--sim_path', type=str, help='environment to load', required= True, default='/../envs/fm/assets/franka_dmanus.xml')
@click.option('-h', '--horizon', type=int, help='#steps in trajectories', default=50)
@click.option('-f', '--frame_skip', type=int, help='frame_skip', default=40)
def main(sim_path, horizon, frame_skip):

    # Process model to use DManus as end effector
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    raw_sim = SimScene.get_sim(curr_dir+sim_path)
    raw_xml = raw_sim.model.get_xml()
    processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
    processed_model_path = curr_dir+sim_path[:-4]+"_processed.xml"
    with open(processed_model_path, 'w') as file:
        file.write(processed_xml)

    # Prep
    sim = SimScene.get_sim(processed_model_path)
    time_horizon = horizon * frame_skip * sim.model.opt.timestep
    dt = sim.model.opt.timestep*frame_skip
    os.remove(processed_model_path)


    # setup
    target_sid = sim.model.site_name2id("workspace")
    ARM_JNT0 = np.mean(sim.model.jnt_range[:ARM_nJnt], axis=-1)

    target_pos = BIN_POS
    target_elr = np.array([-1.57, 0 , -1.57])
    target_quat= euler2quat(target_elr)

    # reseed the arm for IK
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.forward()

    ik_result_up_open = qpos_from_site_pose(
                physics = sim,
                site_name = "end_effector",
                target_pos= target_pos+np.array([0, 0, drop_z]),
                target_quat= target_quat,
                inplace=False,
                regularization_strength=1.0)
    ik_result_up_close = qpos_from_site_pose(
                physics = sim,
                site_name = "end_effector",
                target_pos= target_pos+np.array([0, 0, drop_z]),
                target_quat= target_quat,
                inplace=False,
                regularization_strength=1.0)

    ik_result_up_close.qpos[:] = ik_result_up_open.qpos[:]

    ik_result_up_open.qpos[-10:] = HAND_OPEN
    ik_result_up_close.qpos[-10:] = HAND_CLOSE
    print("IK:: Status:{}, total steps:{}, err_norm:{}".format(ik_result_up_open.success, ik_result_up_open.steps, ik_result_up_open.err_norm))

    while True:

        # Update targets
        if sim.data.time==0:
            print("Resamping new target")

            # sample targets
            target_pos = BIN_POS + np.random.uniform(high=BIN_DIM, low=-1*BIN_DIM) + np.array([0, 0, 0.10]) # add some z offfset
            # target_elr = np.random.uniform(high= [3.14, 0, 0], low=[3.14, 0, -3.14])
            target_elr = np.array([-1.57, 0, -1.57]) + np.random.uniform(high=[0, 0 ,.5], low=[0, 0 ,-.5])
            target_quat= euler2quat(target_elr)
            target_mat = euler2mat(target_elr)

            # propagage targets to the sim for viz
            sim.model.site_pos[target_sid] = target_pos
            # sim.model.site_xmat[target_sid] = target_mat.flatten()

            # reseed the arm for IK
            sim.data.qpos[:ARM_nJnt] = ARM_JNT0
            sim.forward()

            # IK
            ik_result_down = qpos_from_site_pose(
                physics = sim,
                site_name = "end_effector",
                target_pos= target_pos,
                target_quat= target_quat,
                inplace=False,
                regularization_strength=1.0)

            print("IK:: Status:{}, total steps:{}, err_norm:{}".format(ik_result_down.success, ik_result_down.steps, ik_result_down.err_norm))

            # generate min jerk trajectory
            down_open_qpos = ik_result_down.qpos.copy()
            down_open_qpos[-10:] = HAND_OPEN

            down_close_qpos = ik_result_down.qpos.copy()
            down_close_qpos[-10:] = HAND_CLOSE

            waypoints1 =  generate_joint_space_min_jerk(start=ik_result_up_open.qpos[:ARM_nJnt], goal=down_open_qpos, time_to_go=time_horizon, dt=dt)
            waypoints2 = generate_joint_space_min_jerk(start=down_open_qpos, goal=down_close_qpos, time_to_go=time_horizon, dt=dt)

            waypoints3 = generate_joint_space_min_jerk(start=down_close_qpos, goal=ik_result_up_close.qpos[:ARM_nJnt], time_to_go=time_horizon, dt=dt)

            waypoints4 = generate_joint_space_min_jerk(start=ik_result_up_close.qpos[:ARM_nJnt], goal=ik_result_up_open.qpos[:ARM_nJnt], time_to_go=time_horizon, dt=dt)


            waypoints = waypoints1 + waypoints2 + waypoints3 + waypoints4
            paths = []
            path = {}
            wps = tensor_utils.stack_tensor_dict_list(waypoints)
            path['actions'] =  wps['position']
            path['env_infos'] = {
                'obs_dict': {'qp': wps['position']},
                'state':    {'qpos': wps['position'], 'qvel': wps['position']}
                }
            paths.append(path)
            file_name = 'ik_traj.pickle'
            pickle.dump(paths, open(file_name, 'wb'))
            print("Saved: "+file_name)

        # propagate waypoint in sim
        waypoint_ind = int(sim.data.time/dt)

        sim.data.qpos[:ARM_nJnt] = waypoints[waypoint_ind]['position']
        sim.advance(render=True)

        # reset time if time_horizon elapsed
        if sim.data.time>time_horizon*4:
            sim.reset()


if __name__ == '__main__':
    main()
