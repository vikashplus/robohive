# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/mj_envs
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Arm+Gripper tele-op using input devices (keyboard / spacenav) \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

from mj_envs.utils.quat_math import euler2quat, mulQuat
from mj_envs.utils.inverse_kinematics import IKResult, qpos_from_site_pose
import numpy as np
import click
import gym

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer, ignore_mujoco_warnings
except ImportError as e:
    raise ImportError("(HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)")
try:
    from vtils.input.keyboard import KeyInput as KeyBoard
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


@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default='rpFrankaRobotiqData-v0')
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-i', '--input_device', type=click.Choice(['keyboard', 'spacemouse']), help='input to use for teleOp', default='keyboard')
@click.option('-h', '--horizon', type=int, help='Rollout horizon', default=100)
@click.option('-n', '--num_rollouts', type=int, help='number of repeats for the rollouts', default=1)
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
def main(env_name, env_args, input_device, horizon, num_rollouts, seed, goal_site, teleop_site, pos_scale, rot_scale, gripper_scale, vendor_id, product_id):
    # x_range, y_range, z_range, roll_range, pitch_range, yaw_range, gripper_range

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)
    env.env.mujoco_render_frames = True
    goal_sid = env.sim.model.site_name2id(goal_site)
    env.sim.model.site_rgba[goal_sid][3] = 0.2 # make visible

    # prep input device
    if input_device=='keyboard':
        input = KeyBoard()
    elif input_device=='spacemouse':
        input = SpaceMouse(vendor_id=vendor_id, product_id=product_id)

        print("Press both keys to stop listening")
    done = False

    # default actions
    act = np.zeros(env.action_space.shape)
    gripper_state = 0

    # Collect rollout
    for i_rollout in range(num_rollouts):
        env.reset()

        for i_step in range(horizon):

            # prep input device
            if input_device=='keyboard':
                delta_pos, delta_euler, delta_gripper, done = poll_keyboard(input)
            elif input_device=='spacemouse':
                delta_pos, delta_euler, delta_gripper, done = poll_spacemouse(input)

            if done: break

            # udpate pos
            curr_pos = env.sim.model.site_pos[goal_sid]
            curr_pos[:] += pos_scale*delta_pos
            # update rot
            curr_quat =  env.sim.model.site_quat[goal_sid]
            curr_quat[:] = mulQuat(euler2quat(rot_scale*delta_euler), curr_quat)
            # update gripper
            if delta_gripper !=0:
                gripper_state = gripper_scale*delta_gripper # TODO: Update to be delta

            ik_result = qpos_from_site_pose(
                        physics = env.sim,
                        site_name = teleop_site,
                        target_pos= curr_pos,
                        target_quat= curr_quat,
                        inplace=False,
                        regularization_strength=1.0)
            print(f"IK(t:{i_step}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
            act[:7] = ik_result.qpos[:7]
            act[7:] = gripper_state
            if env.normalize_act:
                act = env.env.robot.normalize_actions(act)
            env.step(act)
        if done: break


if __name__ == '__main__':
    main()