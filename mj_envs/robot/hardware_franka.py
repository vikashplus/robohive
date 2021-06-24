from typing import Dict, Sized
import time

import numpy as np
from numpy.core.fromnumeric import size
import torch

from fair_controller_manager import RobotInterface
import torchcontrol as toco
from .hardware_base import hardwareBase
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
        q_current = state_dict["joint_pos"]
        qd_current = state_dict["joint_vel"]

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, self.q_desired, torch.zeros_like(qd_current)
        )

        return {"torque_desired": output}



class FrankaArm(hardwareBase):
    def __init__(self, name, ip_address, **kwargs):
        self.name = name
        self.robot = None

        # Initialize self.robot interface
        self.robot = RobotInterface(
            ip_address=ip_address,
        )
        # self.reset()

    def connect(self, policy=None):
        """Establish hardware connection"""

        if policy==None:
            # Create policy instance
            q_initial = self.get_sensors()
            default_kq = .025*torch.Tensor(self.robot.metadata.default_Kq)
            default_kqd = .025*torch.Tensor(self.robot.metadata.default_Kqd)
            policy = JointPDPolicy(
                desired_joint_pos=q_initial,
                kq=default_kq,
                kqd=default_kqd,
            )

        # Send policy
        print("\nRunning PD policy...")
        self.robot.send_torch_policy(policy, blocking=False)

    def okay(self):
        """Return hardware health"""
        if self.robot:
            return True
        else:
            return False

    def close(self):
        """Close hardware connection"""
        print("Terminating PD policy...")
        state_log = True
        if self.robot:
            self.reset()
            state_log = self.robot.terminate_current_policy()
        return state_log

    def reset(self):
        """Reset hardware"""
        self.robot.go_home()

    def get_sensors(self):
        """Get hardware sensors"""
        joint_angel = self.robot.get_joint_angles()
        return joint_angel

    def apply_commands(self, q_desired):
        """Apply hardware commands"""
        q_initial = self.get_sensors()
        q_des_tensor = torch.tensor(q_desired)
        # print(q_initial, q_des_tensor)
        self.robot.update_current_policy({"q_desired": q_des_tensor})

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
    franka.connect(policy=None)

    # Update policy to execute a sine trajectory on joint 6 for 5 seconds
    print("Starting sine motion updates...")
    q_initial = franka.get_sensors()
    q_desired = q_initial.clone()
    print(list(q_desired.shape))

    for i in range(int(time_to_go * hz)):
        # q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
        # q_desired[5] = q_initial[5] + 0.05*np.random.uniform(high=1, low=-1)
        q_desired = q_initial + 0.01*np.random.uniform(high=1, low=-1, size=7)

        franka.apply_commands(q_desired = q_desired)
        time.sleep(1 / hz)

    franka.close()

