# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/mj_envs
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Arm+Gripper tele-op using oculus \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

import time
import numpy as np
import click
import gym
from mj_envs.utils.quat_math import euler2quat, euler2mat, mat2quat, diffQuat, mulQuat
from mj_envs.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from mj_envs.logger.grouped_datasets import Trace

try:
    from oculus_reader import OculusReader
except ImportError as e:
    raise ImportError("(Missing oculus_reader. HINT: Install and perform the setup instructions from https://github.com/rail-berkeley/oculus_reader)")

# VR ==> MJ mapping when teleOp user is standing infront of the robot
def vrfront2mj(pose):
    pos = np.zeros([3])
    pos[0] = -1.*pose[2][3]
    pos[1] = -1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.*pose[2][:3]
    mat[1][:] = +1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    return pos, mat2quat(mat)

# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = np.zeros([3])
    pos[0] = +1.*pose[2][3]
    pos[1] = +1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = +1.*pose[2][:3]
    mat[1][:] = -1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    return pos, mat2quat(mat)

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default='rpFrankaRobotiqData-v0')
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-h', '--horizon', type=int, help='Rollout horizon', default=100)
@click.option('-n', '--num_rollouts', type=int, help='number of repeats for the rollouts', default=2)
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='Where to render?', default='onscreen')
@click.option('-gs', '--goal_site', type=str, help='Site that updates as goal using inputs', default='ee_target')
@click.option('-ts', '--teleop_site', type=str, help='Site used for teleOp/target for IK', default='end_effector')
@click.option('-ps', '--pos_scale', type=float, default=0.05, help=('position scaling factor'))
@click.option('-rs', '--rot_scale', type=float, default=0.1, help=('rotation scaling factor'))
@click.option('-gs', '--gripper_scale', type=float, default=1, help=('gripper scaling factor'))
# @click.option('-tx', '--x_range', type=tuple, default=(-0.5, 0.5), help=('x range'))
# @click.option('-ty', '--y_range', type=tuple, default=(-0.5, 0.5), help=('y range'))
# @click.option('-tz', '--z_range', type=tuple, default=(-0.5, 0.5), help=('z range'))
# @click.option('-rx', '--roll_range', type=tuple, default=(-0.5, 0.5), help=('roll range'))
# @click.option('-ry', '--pitch_range', type=tuple, default=(-0.5, 0.5), help=('pitch range'))
# @click.option('-rz', '--yaw_range', type=tuple, default=(-0.5, 0.5), help=('yaw range'))
# @click.option('-gr', '--gripper_range', type=tuple, default=(0, 1), help=('z range'))
def main(env_name, env_args, horizon, num_rollouts, seed, render, goal_site, teleop_site, pos_scale, rot_scale, gripper_scale):
    # x_range, y_range, z_range, roll_range, pitch_range, yaw_range, gripper_range):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)
    if render=='onscreen':
        env.env.mujoco_render_frames = True
    else:
        env.env.mujoco_render_frames = False

    goal_sid = env.sim.model.site_name2id(goal_site)
    teleop_sid = env.sim.model.site_name2id(teleop_site)
    env.sim.model.site_rgba[goal_sid][3] = 0.2 # make visible

    # prep input device
    oculus_reader = OculusReader()
    pos_offset = env.sim.model.site_pos[goal_sid].copy()
    quat_offset = env.sim.model.site_quat[goal_sid].copy()
    oculus_reader_ready = False
    while not oculus_reader_ready:
        transformations, buttons = oculus_reader.get_transformations_and_buttons()
        if transformations or buttons:
            oculus_reader_ready = True
        else:
            print("Oculus reander not ready. Check that headset is awake and controller are on")
        time.sleep(0.10)

    trace = Trace("TeleOp Trajectories")

    # default actions
    act = np.zeros(env.action_space.shape)
    gripper_state = delta_gripper = 0

    # Collect rollouts
    env.reset(blocking=False) # blocking reset to prep the scene
    for i_rollout in range(num_rollouts):

        # start a new rollout
        print("rollout {} start".format(i_rollout))
        group_key='Trial'+str(i_rollout); trace.create_group(group_key)
        rollout_done = False
        # Reset goal back to nominal position
        env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]

        # Gather single rollout
        for i_step in range(horizon):

            # Pool device readings
            transformations, buttons = oculus_reader.get_transformations_and_buttons()

            # Check for reset request
            if buttons and buttons['B']:
                env.sim.model.site_pos[goal_sid] =  pos_offset
                env.sim.model.site_quat[goal_sid] =  quat_offset
                rollout_done = True

            if not rollout_done:
                # Process request
                if transformations and 'r' in transformations:
                    right_controller_pose = transformations['r']
                    # VRpos, VRquat = vrfront2mj(right_controller_pose)
                    VRpos, VRquat = vrbehind2mj(right_controller_pose)

                    # Update targets if engaged
                    if buttons['RG']:
                        # dVRP/R = VRP/Rt - VRP/R0
                        dVRP = VRpos - VRP0
                        # dVRR = VRquat - VRR0
                        dVRR = diffQuat(VRR0, VRquat)
                        # MJP/Rt =  MJP/R0 + dVRP/R
                        env.sim.model.site_pos[goal_sid] = MJP0 + dVRP
                        env.sim.model.site_quat[goal_sid] = mulQuat(MJR0, dVRR)
                        delta_gripper = buttons['rightTrig'][0]

                    # Adjust origin if not engaged
                    else:
                        # RP/R0 = RP/Rt
                        MJP0 = env.sim.model.site_pos[goal_sid].copy()
                        MJR0 = env.sim.model.site_quat[goal_sid].copy()

                        # VP/R0 = VP/Rt
                        VRP0 = VRpos
                        VRR0 = VRquat

                    # udpate desired pos
                    target_pos = env.sim.model.site_pos[goal_sid]
                    # target_pos[:] += pos_scale*delta_pos
                    # update desired orientation
                    target_quat =  env.sim.model.site_quat[goal_sid]
                    # target_quat[:] = mulQuat(euler2quat(rot_scale*delta_euler), target_quat)
                    # update desired gripper
                    gripper_state = gripper_scale*delta_gripper # TODO: Update to be delta

                    # Find joint space solutions
                    ik_result = qpos_from_site_pose(
                                physics = env.sim,
                                site_name = teleop_site,
                                target_pos= target_pos,
                                target_quat= target_quat,
                                inplace=False,
                                regularization_strength=1.0)

                    # Command robot
                    if ik_result.success:
                        act[:7] = ik_result.qpos[:7]
                        act[7:] = gripper_state
                        if env.normalize_act:
                            act = env.env.robot.normalize_actions(act)
                        obs, rwd, done, env_info = env.step(act)

                        # log values
                        datum_dict = dict(
                                time=i_step,
                                observation=obs,
                                action=act,
                                reward=rwd,
                                env_info=env_info,
                                done=done,
                            )
                        trace.append_datums(group_key=group_key,
                            dataset_key_val=datum_dict)

                        # Detect jumps
                        qpos_now = env_info['obs_dict']['qp_arm']
                        qpos_arm_err = np.linalg.norm(ik_result.qpos[:7]-qpos_now[:7])
                        if qpos_arm_err>0.5:
                            print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
                            # Reset goal back to nominal position
                            env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
                            env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))
                    else:
                        print(f"IK(t:{i_step}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
            else:
                print("Rollout done. ")
                # user = input("Save rollout?")

                # Reset env back to nominal and break out of this rollout gather
                env.reset(blocking=False)
                break

        print("rollout {} end".format(i_rollout))
        time.sleep(0.5)

    trace.save("teleOp_trace.h5", verify_length=True)
    oculus_reader.stop()
    env.close()


if __name__ == '__main__':
    main()