# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/mj_envs
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Calculate min jerk trajectory using IK \n
    - NOTE: written for franka_busbin_v0.xml model and might not be too generic
EXAMPLE:\n
    - python tutorials/get_ik_minjerk_trajectory.py --sim_path envs/arms/franka/assets/franka_busbin_v0.xml\n
"""

from mujoco_py import load_model_from_path, MjSim, MjViewer

from mj_envs.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from mj_envs.utils.min_jerk import *
from mj_envs.utils.quat_math import euler2quat, euler2mat
import click
import numpy as np

BIN_POS = np.array([-.235, 0.5, .85])
BIN_DIM = np.array([.2, .3, 0])
ARM_nJnt = 7

@click.command(help=DESC)
@click.option('-s', '--sim_path', type=str, help='environment to load', required= True)
@click.option('-h', '--horizon', type=int, help='time (s) to simulate', default=2)
def main(sim_path, horizon):
    # Prep
    model = load_model_from_path(sim_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    # setup
    target_sid = sim.model.site_name2id("drop_target")
    ARM_JNT0 = np.mean(sim.model.jnt_range[:ARM_nJnt], axis=-1)

    while True:

        # Update targets
        if sim.data.time==0:
            print("Resamping new target")

            # sample targets
            target_pos = BIN_POS + np.random.uniform(high=BIN_DIM, low=-1*BIN_DIM) + np.array([0, 0, 0.10]) # add some z offfset
            target_elr = np.random.uniform(high= [3.14, 0, 0], low=[3.14, 0, -3.14])
            target_quat= euler2quat(target_elr)
            target_mat = euler2mat(target_elr)

            # propagage targets to the sim for viz
            sim.data.site_xpos[target_sid] = target_pos
            sim.data.site_xmat[target_sid] = target_mat.flatten()

            # reseed the arm for IK
            sim.data.qpos[:ARM_nJnt] = ARM_JNT0
            sim.forward()

            # IK
            ik_result = qpos_from_site_pose(
                physics = sim,
                site_name = "end_effector",
                target_pos= target_pos,
                target_quat= target_quat,
                inplace=False,
                regularization_strength=1.0)
            print("IK:: Status:{}, total steps:{}, err_norm:{}".format(ik_result.success, ik_result.steps, ik_result.err_norm))

            # generate min jerk trajectory
            waypoints =  generate_joint_space_min_jerk(start=ARM_JNT0, goal=ik_result.qpos[:ARM_nJnt], time_to_go=horizon, dt=sim.model.opt.timestep )

        # propagate waypoint in sim
        waypoint_ind = int(sim.data.time/sim.model.opt.timestep)
        sim.data.qpos[:ARM_nJnt] = waypoints[waypoint_ind]['position']
        sim.forward()

        # update time and render
        sim.data.time += sim.model.opt.timestep
        viewer.render()

        # reset time if horizon elapsed
        if sim.data.time>horizon:
            sim.data.time = 0

if __name__ == '__main__':
    main()
