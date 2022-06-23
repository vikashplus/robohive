""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from typing import Dict, Sized
import time

import numpy as np
from numpy.core.fromnumeric import size
import torch

from polymetis import RobotInterface
import torchcontrol as toco
from mj_envs.robot.hardware_base import hardwareBase
import argparse

class JointPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, desired_joint_pos, kq, kqd, **kwargs):
        """
        Args:
            desired_joint_pos (int):    Number of steps policy should execute
            hz (double):                Frequency of controller
            kq, kqd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.q_desired = torch.nn.Parameter(desired_joint_pos)

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, self.q_desired, torch.zeros_like(qd_current)
        )

        return {"joint_torques": output}



class FrankaArm(hardwareBase):
    def __init__(self, name, ip_address, **kwargs):
        self.name = name
        self.ip_address = ip_address
        self.robot = None

    def connect(self, policy=None):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = RobotInterface(
                ip_address=self.ip_address,
                enforce_version=False
            )
            print("Success")
        except Exception as e:
            self.robot = None # declare dead
            print("Failed with exception: ", e)
            return connection

        print("Testing {} connection: ".format(self.name), end="")
        connection = self.okay()
        if connection:
            print("okay")
            state = self.robot.get_robot_state()
            if policy==None:
                # Create policy instance
                s_initial = self.get_sensors()
                default_kq = 0.10*torch.Tensor(self.robot.metadata.default_Kq)
                default_kqd = 0.10*torch.Tensor(self.robot.metadata.default_Kqd)
                policy = JointPDPolicy(
                    desired_joint_pos=s_initial['joint_pos'],
                    kq=default_kq,
                    kqd=default_kqd,
                )

            # Send policy
            print("\nRunning PD policy...")
            self.robot.send_torch_policy(policy, blocking=False)
        else:
            print("Not ready. Please retry connection")

        return connection

    def okay(self):
        """Return hardware health"""
        okay = False
        if self.robot:
            try:
                state = self.robot.get_robot_state()
                delay = time.time() - (state.timestamp.seconds + 1e-9 * state.timestamp.nanos)
                assert delay < 5, "Acquired state is stale by {} seconds".format(delay)
                okay = True
            except:
                self.robot = None # declare dead
                okay = False
        return okay

    def close(self):
        """Close hardware connection"""
        if self.robot:
            print("Terminating PD policy: ", end="")
            try:
                state_log = self.robot.terminate_current_policy()
                print("Success")
            except:
                print("Failed. Resetting directly to home: ", end="")
            self.reset()
            self.robot = None
            print("Done")
        return True


    def reconnect(self):
        print("Attempting re-connection")
        self.connect()
        while not self.okay():
            self.connect()
            time.sleep(2)
        print("Re-connection success")


    def reset(self, time_to_go=5):
        """Reset hardware"""
        self.robot.go_home(time_to_go=time_to_go)


    def get_sensors(self):
        """Get hardware sensors"""
        try:
            joint_pos = self.robot.get_joint_positions()
            joint_vel = self.robot.get_joint_velocities()
        except:
            print("Failed to get current sensors: ", end="")
            self.reconnect()
            return self.get_sensors()
        return {'joint_pos': joint_pos, 'joint_vel':joint_pos}


    def apply_commands(self, q_desired):
        """Apply hardware commands"""
        q_des_tensor = torch.tensor(q_desired)
        try:
            self.robot.update_current_policy({"q_desired": q_des_tensor})
        except Exception as e:
            print("1> Failed to udpate policy with exception", e)
            self.reconnect()

    def __del__(self):
        self.close()


# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="OptiTrack Client: Connects to \
        the server and fetches streaming data")

    parser.add_argument("-i", "--server_ip",
                        type=str,
                        help="IP address or hostname of the franka server",
                        default="localhost") # 10.0.0.123 # "169.254.163.91",
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # user inputs
    time_to_go = 2*np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    franka = FrankaArm(name="Franka-Demo", ip_address=args.server_ip)

    # connect to robot with default policy
    assert franka.connect(policy=None), "Connection to robot failed."

    # Update policy to execute a sine trajectory on joint 6 for 5 seconds
    print("Starting sine motion updates...")
    s_initial = franka.get_sensors()
    q_desired = s_initial['get_sensors'].clone()
    print(list(q_desired.shape))

    for i in range(int(time_to_go * hz)):
        q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
        # q_desired[5] = q_initial[5] + 0.05*np.random.uniform(high=1, low=-1)
        # q_desired = q_initial + 0.01*np.random.uniform(high=1, low=-1, size=7)

        franka.apply_commands(q_desired = q_desired)
        time.sleep(1 / hz)

    franka.close()
