from typing import Dict, Sized
import time

import numpy as np
from numpy.core.fromnumeric import size
import torch

from polymetis import RobotInterface
import torchcontrol as toco
from hardware_base import hardwareBase
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
    CMD_SHAPE = 7
    # Notice that the start position will add joint offset before send to hdware
    START_POSITION = np.array([-0.145, -0.67, -0.052, -2.3, 0.145, 1.13, 0.029])
    # Notice that the joint limit is applied after applying joint offset
    JOINT_LIMIT_MIN = np.array(
        [-2.8773, -1.7428, -2.8773, -3.0018, -2.8773, 0.0025, -2.8773])
    JOINT_LIMIT_MAX = np.array(
        [2.8773, 1.7428, 2.8773, -0.1398, 2.8773, 3.7325, 2.8773])

    def __init__(self, name, ip_address, **kwargs):
        self.name = name
        self.robot = None
        self.JOINT_OFFSET = np.array(
            [0, 0, 0, 0, 0., np.pi/2, np.pi/4],
            dtype=np.float32) # TODO replace hardcode with config

        # Initialize self.robot interface
        self.robot = RobotInterface(
            ip_address=ip_address,
            enforce_version=False,
        )

    def default_policy(self, kq_ratio=1.5, kqd_ratio=1.5):
        q_initial = self.get_sensors().clone()
        kq = kq_ratio * torch.Tensor(self.robot.metadata.default_Kq)
        kqd = kqd_ratio * torch.Tensor(self.robot.metadata.default_Kqd)
        return JointPDPolicy(
                desired_joint_pos=torch.tensor(q_initial),
                kq=kq, kqd=kqd,
        )

    def connect(self, policy=None):
        """Establish hardware connection"""
        if policy==None:
            print("\nRunning PD policy...")
            policy = self.default_policy()
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

    # def get_sensors(self):
    #     """Get hardware sensors"""
    #     return self.robot.get_joint_angles()

    # def apply_commands(self, q_desired):
    #     """Apply hardware commands"""
    #     q_des_tensor = torch.tensor(q_desired)
    #     self.robot.update_current_policy({"q_desired": q_des_tensor})

    def get_sensors(self):
        """Get hardware sensors (apply offset)"""
        joint_angle = self.robot.get_joint_angles()
        return joint_angle - self.JOINT_OFFSET

    def apply_commands(self, q_desired):
        """Apply hardware commands (apply offset)"""
        q_des_tensor = np.array(q_desired) + self.JOINT_OFFSET
        q_des_tensor = torch.tensor(np.clip(
            q_des_tensor, self.JOINT_LIMIT_MIN, self.JOINT_LIMIT_MAX))
        #print('RAW_CMD_DIFF', q_des_tensor.float() - self.robot.get_joint_angles())
        self.robot.update_current_policy({"q_desired": q_des_tensor.float()})

    def __del__(self):
        self.close()

from hardware_gripper import Gripper

class FrankaArmWithGripper(FrankaArm):
    CMD_SHAPE = 8
    START_POSITION = np.array(list(FrankaArm.START_POSITION) + [0.08], dtype=np.float32)

    def __init__(self, name, ip_address, **kwargs):
        super(FrankaArmWithGripper, self).__init__(name, ip_address, **kwargs)
        self.gripper = Gripper(ip_address)
        self.state_nparray = np.zeros(self.CMD_SHAPE)

    def reset(self):
        self.open_gripper()
        super(FrankaArmWithGripper, self).reset()

    def close_gripper(self):
        self.gripper.goto(width=0.0, speed=0.1, force=0.1)

    def open_gripper(self):
        self.gripper.goto(width=self.gripper.gripper_max_width, speed=0.1, force=0.1)

    def get_sensors(self):
        """Get hardware sensors (apply offset)"""
        self.state_nparray[0:7] = super(FrankaArmWithGripper, self).get_sensors()
        self.state_nparray[-1] = self.gripper.get()
        return self.state_nparray

    def apply_commands(self, q_desired):
        """Apply hardware commands (apply offset)"""
        super(FrankaArmWithGripper, self).apply_commands(q_desired[:-1])
        self.gripper.goto(width=q_desired[-1], speed=0.1, force=0.1)

class FrankaArmWithRobotiQGripper(FrankaArmWithGripper):

    def __init__(self, name, ip_address, **kwargs):
        super(FrankaArmWithRobotiQGripper, self).__init__(name, ip_address, **kwargs)
        self.JOINT_OFFSET[6] -= np.pi/4


# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="OptiTrack Client: Connects to \
        the server and fetches streaming data")

    parser.add_argument("-i", "--server_ip",
                        type=str,
                        help="IP address or hostname of the franka server",
                        default="172.16.0.1") # 10.0.0.123 # "169.254.163.91",
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # user inputs
    time_to_go = 2*np.pi
    joint = 6
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    franka = FrankaArm(name="Franka-Demo", ip_address=args.server_ip)

    import time
    start = time.time()
    franka.robot.go_home()
    # connect to robot with default policy
    franka.connect(policy=None)

    print(f"Took {time.time() - start} sec to go home and set policy")

    q_initial = franka.get_sensors()
    q_desired = np.array(q_initial, dtype=np.float64)
    print(f"Shape of q_desired: {list(q_desired.shape)}")
    print(f"Starting sine motion updates... will repeat for {time_to_go} seconds")

    for i in range(int(time_to_go * hz)):
        q_desired[joint] = q_initial[joint] + m * np.sin(np.pi * i / (T * hz))
        franka.apply_commands(q_desired = q_desired)
        time.sleep(1 / hz)

    print("Finished moving")
    franka.close()