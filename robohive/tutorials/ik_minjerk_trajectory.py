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
    - python tutorials/ik_minjerk_trajectory.py --sim_path envs/arms/franka/assets/franka_busbin_v0.xml\n
"""

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import *
from robohive.utils.quat_math import euler2quat
import click
import numpy as np

BIN_POS = np.array([.235, 0.5, .85])
BIN_DIM = np.array([.2, .3, 0])
BIN_TOP = 0.10
ARM_nJnt = 7

@click.command(help=DESC)
@click.option('-s', '--sim_path', type=str, help='environment to load', required= True, default='envs/arms/franka/assets/franka_busbin_v0.xml')
@click.option('-h', '--horizon', type=int, help='time (s) to simulate', default=2)
def main(sim_path, horizon):
    # Prep
    sim = SimScene.get_sim(model_handle=sim_path)

    # setup
    target_sid = sim.model.site_name2id("drop_target")
    ARM_JNT0 = np.mean(sim.model.jnt_range[:ARM_nJnt], axis=-1)

    while True:

        # Update targets
        if sim.data.time==0:
            print("Resamping new target")

            # sample targets
            target_pos = BIN_POS + np.random.uniform(high=BIN_DIM, low=-1*BIN_DIM) + np.array([0, 0, BIN_TOP]) # add some z offfset
            target_elr = np.random.uniform(high= [3.14, 0, 0], low=[3.14, 0, -3.14])
            target_quat= euler2quat(target_elr)

            # propagage targets to the sim for viz
            sim.model.site_pos[target_sid][:] = target_pos - np.array([0, 0, BIN_TOP])
            sim.model.site_quat[target_sid][:] = target_quat

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
        sim.data.ctrl[:ARM_nJnt] = waypoints[waypoint_ind]['position']
        sim.advance(render=True)

        # reset time if horizon elapsed
        if sim.data.time>horizon:
            sim.reset()

if __name__ == '__main__':
    main()
