# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Arm+Gripper tele-op using input devices (keyboard / spacenav) \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

from robohive.utils.quat_math import euler2quat, mulQuat
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.logger.roboset_logger import RoboSet_Trace
from robohive.logger.grouped_datasets import Trace as RoboHive_Trace
import numpy as np
import click
from robohive.utils import gym

try:
    from vtils.input.keyboard import KeyInput as KeyBoard
    from vtils.input.gamepad import GamePad
    from vtils.input.spacemouse import SpaceMouse
except ImportError as e:
    raise ImportError("Please install vtils -- https://github.com/vikashplus/vtils")

# Poll and process keyboard values
def poll_keyboard(input_device):
    # get sensors
    sen = input_device.get_sensors()

    # exit request
    done = True if sen=='esc' else False

    # gripper
    if sen == '[':
        delta_gripper = -1
    elif sen == ']':
        delta_gripper = 1
    else:
        delta_gripper = 0

    # positions
    delta_pos = np.array([0, 0, 0])
    if sen == 'd':
        delta_pos[0] = 1
    elif sen == 'a':
        delta_pos[0] = -1
    elif sen == 'w':
        delta_pos[1] = 1
    elif sen == 'x':
        delta_pos[1] = -1
    elif sen == 'q':
        delta_pos[2] = 1
    elif sen == 'z':
        delta_pos[2] = -1

    # rotations
    delta_euler = np.array([0, 0, 0])
    if sen == 'up':
        delta_euler[0] = -1
    elif sen == 'down':
        delta_euler[0] = 1
    elif sen == 'left':
        delta_euler[1] = -1
    elif sen == 'right':
        delta_euler[1] = 1
    elif sen == ',':
        delta_euler[2] = -1
    elif sen == '.':
        delta_euler[2] = 1

    return delta_pos, delta_euler, delta_gripper, done


# Poll and process spacemouse values
def poll_spacemouse(input_device):
    # get sensors
    sen = input_device.get_sensors()

    # exit request
    done = True if (sen['left'] and sen['right']) else False

    # gripper
    if sen['left'] == True:
        delta_gripper = -1
    elif sen['right'] == True:
        delta_gripper = 1
    else:
        delta_gripper = 0

    # positions
    delta_pos = np.array([sen['x'], sen['y'], sen['z']])

    # rotations
    delta_euler = np.array([sen['roll'], sen['pitch'], sen['yaw']])

    return delta_pos, delta_euler, delta_gripper, done


# Poll and process gamepad values
def poll_gamepad(input_device):
    # get sensors
    sen = input_device.get_sensors()

    # exit request
    done = True if (sen["BTN_START"] and sen["BTN_SELECT"]) else False

    scale_factor = 1.0

    if sen["BTN_NORTH"] == 1:
        scale_factor = 0.25 # Hold X to slow down arm movement
    
    if sen["BTN_WEST"] == 1: # B: open gripper
        delta_gripper = 1
    elif sen["BTN_EAST"] == 1: # Y: close gripper
        delta_gripper = -1
    else:
        delta_gripper = 0

    # positions
    delta_pos = np.array([0, 0, 0])
    # Moving EE forward or backward, when facing the robot
    delta_pos[0] = sen["ABS_Y"]
    # Moving EE left or right, when facing the robot
    delta_pos[1] = sen["ABS_X"]
    # Raise or lower
    delta_pos[2] = sen["ABS_Z"] - sen["ABS_RZ"]

    # rotations
    delta_euler = np.array([0, 0, 0])
    if sen["ABS_HAT0X"] == -1:
        delta_euler[0] = -1
    elif sen["ABS_HAT0X"] == 1:
        delta_euler[0] = 1
    elif sen["ABS_HAT0Y"] == 1:
        delta_euler[1] = -1
    elif sen["ABS_HAT0Y"] == -1:
        delta_euler[1] = 1
    elif sen["BTN_TL"] == 1:
        delta_euler[2] = -1
    elif sen["BTN_TR"] == 1:
        delta_euler[2] = 1

    return delta_pos * scale_factor, delta_euler * scale_factor, delta_gripper, done


@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default='rpFrankaRobotiqData-v0')
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-rn', '--reset_noise', type=float, default=0.0, help=('Amplitude of noise during reset'))
@click.option('-an', '--action_noise', type=float, default=0.0, help=('Amplitude of action noise during rollout'))
@click.option('-i', '--input_device', type=click.Choice(['keyboard', 'spacemouse', 'gamepad']), help='input to use for teleOp', default='keyboard')
@click.option('-o', '--output', type=str, default="teleOp_trace.h5", help=('Output name'))
@click.option('-h', '--horizon', type=int, help='Rollout horizon', default=100)
@click.option('-n', '--num_rollouts', type=int, help='number of repeats for the rollouts', default=1)
@click.option('-f', '--output_format', type=click.Choice(['RoboHive', 'RoboSet']), help='Data format', default='RoboHive')
@click.option('-c', '--camera', multiple=True, type=str, default=[], help=('list of camera topics for rendering'))
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'onscreen+offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-gs', '--goal_site', type=str, help='Site that updates as goal using inputs', default='ee_target')
@click.option('-ts', '--teleop_site', type=str, help='Site used for teleOp/target for IK', default='end_effector')
@click.option('-ps', '--pos_scale', type=float, default=0.05, help=('position scaling factor'))
@click.option('-rs', '--rot_scale', type=float, default=0.1, help=('rotation scaling factor'))
@click.option('-gs', '--gripper_scale', type=float, default=1, help=('gripper scaling factor'))
@click.option('-vi', '--vendor_id', type=int, default=9583, help=('Spacemouse vendor id'))
@click.option('-pi', '--product_id', type=int, default=50741, help=('Spacemouse product id'))
# @click.option('-tx', '--x_range', type=tuple, default=(-0.5, 0.5), help=('x range'))
# @click.option('-ty', '--y_range', type=tuple, default=(-0.5, 0.5), help=('y range'))
# @click.option('-tz', '--z_range', type=tuple, default=(-0.5, 0.5), help=('z range'))
# @click.option('-rx', '--roll_range', type=tuple, default=(-0.5, 0.5), help=('roll range'))
# @click.option('-ry', '--pitch_range', type=tuple, default=(-0.5, 0.5), help=('pitch range'))
# @click.option('-rz', '--yaw_range', type=tuple, default=(-0.5, 0.5), help=('yaw range'))
# @click.option('-gr', '--gripper_range', type=tuple, default=(0, 1), help=('z range'))
def main(env_name, env_args, reset_noise, action_noise, input_device, output, horizon, num_rollouts, output_format, camera, render, seed, goal_site, teleop_site, pos_scale, rot_scale, gripper_scale, vendor_id, product_id):
    # x_range, y_range, z_range, roll_range, pitch_range, yaw_range, gripper_range

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)
    env.env.mujoco_render_frames = True if 'onscreen'in render else False
    goal_sid = env.sim.model.site_name2id(goal_site)
    env.sim.model.site_rgba[goal_sid][3] = 0.2 # make visible

    # prep input device
    if input_device=='keyboard':
        input = KeyBoard()
    elif input_device=='gamepad':
        input = GamePad()
    elif input_device=='spacemouse':
        input = SpaceMouse(vendor_id=vendor_id, product_id=product_id)
        print("Press both keys to stop listening")

    # prep the logger
    if output_format=="RoboHive":
        trace = RoboHive_Trace("TeleOp Trajectories")
    elif output_format=="RoboSet":
        trace = RoboSet_Trace("TeleOp Trajectories")

    # Collect rollouts
    for i_rollout in range(num_rollouts):

        # start a new rollout
        print("rollout {} start".format(i_rollout))
        group_key='Trial'+str(i_rollout); trace.create_group(group_key)
        reset_noise = reset_noise*np.random.uniform(low=-1, high=1, size=env.init_qpos.shape)
        env.reset(reset_qpos=env.init_qpos+reset_noise, blocking=True)

        # recover init state
        obs, rwd, done, env_info = env.forward()
        act = np.zeros(env.action_space.shape)
        gripper_state = 0

        # start rolling out
        for i_step in range(horizon+1):

            # poll input device --------------------------------------
            if input_device=='keyboard':
                delta_pos, delta_euler, delta_gripper, exit_request = poll_keyboard(input)
            elif input_device=='gamepad':
                delta_pos, delta_euler, delta_gripper, exit_request = poll_gamepad(input)
            elif input_device=='spacemouse':
                delta_pos, delta_euler, delta_gripper, exit_request = poll_spacemouse(input)
            if exit_request:
                print("Rollout done. ")
                break

            # recover actions using input ----------------------------
            # udpate pos
            curr_pos = env.sim.model.site_pos[goal_sid]
            curr_pos[:] += pos_scale*delta_pos
            # update rot
            curr_quat =  env.sim.model.site_quat[goal_sid]
            curr_quat[:] = mulQuat(euler2quat(rot_scale*delta_euler), curr_quat)
            # update gripper
            if delta_gripper !=0:
                gripper_state = gripper_scale*delta_gripper # TODO: Update to be delta

            # get action using IK
            ik_result = qpos_from_site_pose(
                        physics = env.sim,
                        site_name = teleop_site,
                        target_pos= curr_pos,
                        target_quat= curr_quat,
                        inplace=False,
                        regularization_strength=1.0)
            if ik_result.success==False:
                print(f"IK(t:{i_step}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
            else:
                act[:7] = ik_result.qpos[:7]
                act[7:] = gripper_state
                if action_noise:
                    act = act + env.env.np_random.uniform(high=action_noise, low=-action_noise, size=len(act)).astype(act.dtype)
                if env.normalize_act:
                    act = env.env.robot.normalize_actions(act)

            # nan actions for last log entry
            act = np.nan*np.ones(env.action_space.shape) if i_step == horizon else act

            # log values at time=t ----------------------------------
            datum_dict = dict(
                    time=env.time,
                    observations=obs,
                    actions=act.copy(),
                    rewards=rwd,
                    env_infos=env_info,
                    done=done,
                )
            trace.append_datums(group_key=group_key,dataset_key_val=datum_dict)
            # print(f't={env.time:2.2}, a={act}, o={obs[:3]}')

            # step env using action from t=>t+1 ----------------------
            if i_step < horizon: #incase last actions (nans) can cause issues in step
                obs, rwd, done, env_info = env.step(act)

        print("rollout {} end".format(i_rollout))

    # save and close
    env.close()
    trace.save(output, verify_length=True)

    # render video outputs
    if len(camera)>0:
        if camera[0]!="default":
            trace.render(output_dir=".", output_format="mp4", groups=":", datasets=camera, input_fps=1/env.dt)
        elif output_format=="RoboHive":
            trace.render(output_dir=".", output_format="mp4", groups=":", datasets=["env_infos/obs_dict/rgb:left_cam:240x424:2d","env_infos/obs_dict/rgb:right_cam:240x424:2d","env_infos/obs_dict/rgb:top_cam:240x424:2d","env_infos/obs_dict/rgb:Franka_wrist_cam:240x424:2d"], input_fps=1/env.dt)
        elif output_format=="RoboSet":
            trace.render(output_dir=".", output_format="mp4", groups=":", datasets=["data/rgb_left","data/rgb_right","data/rgb_top","data/rgb_wrist"], input_fps=1/env.dt)


if __name__ == '__main__':
    main()