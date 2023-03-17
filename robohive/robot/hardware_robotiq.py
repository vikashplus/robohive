from enum import Flag
from polymetis import GripperInterface
from robohive.robot.hardware_base import hardwareBase

import numpy as np
import argparse
import time

class Robotiq(hardwareBase):
    def __init__(self, name, ip_address, **kwargs):
        self.name = name
        self.ip_address = ip_address
        self.robot = None
        self.max_width = 0.0
        self.min_width = 0.0

    def connect(self, policy=None):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("RBQ:> Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = GripperInterface(
                ip_address=self.ip_address,
            )
            print("Success")
        except Exception as e:
            self.robot = None # declare dead
            print("Failed with exception: ", e)
            return connection

        print("RBQ:> Testing {} connection: ".format(self.name), end="")
        if self.okay():
            print("Okay")
            # get max_width based on polymetis version
            if self.robot.metadata:
                self.max_width = self.robot.metadata.max_width
            elif self.robot.get_state().max_width:
                self.max_width = self.robot.get_state().max_width
            else:
                self.max_width = 0.085
            connection = True
        else:
            print("Not ready. Please retry connection")

        return connection

    def okay(self):
        """Return hardware health"""
        okay = False
        if self.robot:
            try:
                state = self.robot.get_state()
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
            print("RBQ:> Resetting robot before close: ", end="")
            try:
                self.reset()
                print("RBQ:> Success: ", end="")
            except:
                print("RBQ:> Failed. Exiting : ", end="")
            self.robot = None
            print("Connection closed")
        return True

    def reconnect(self):
        print("RBQ:> Attempting re-connection")
        self.connect()
        while not self.okay():
            self.connect()
            time.sleep(2)
        print("RBQ:> Re-connection success")


    def reset(self, width=None, **kwargs):
        """Reset hardware"""
        if not width:
            width = self.max_width
        self.apply_commands(width=width, **kwargs)


    def get_sensors(self):
        """Get hardware sensors"""
        try:
            curr_state = self.robot.get_state()
        except:
            print("RBQ:> Failed to get current sensors: ", end="")
            self.reconnect()
            return self.get_sensors()
        return np.array([curr_state.width])

    def apply_commands(self, width:float, speed:float=0.1, force:float=0.1):
        assert width>=0.0 and width<=self.max_width, "Gripper desired width ({}) is out of bound (0,{})".format(width, self.max_width)
        self.robot.goto(width=width, speed=speed, force=force)
        return 0



# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Polymetis based gripper client")

    parser.add_argument("-i", "--server_ip",
                        type=str,
                        help="IP address or hostname of the franka server",
                        default="localhost") # 172.16.0.1

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # user inputs
    time_to_go = 2.0*np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    rbq = Robotiq(name="Demo_robotiq", ip_address=args.server_ip)

    # connect to robot
    status = rbq.connect()
    assert status, "Can't connect to Robotiq"

    # reset using the user controller
    rbq.reset()

    # Close gripper
    des_width = 0.0
    rbq.apply_commands(width=des_width)
    time.sleep(2)
    curr_width = rbq.get_sensors()
    print("RBQ:> Testing gripper close: Desired:{}, Achieved:{}".format(des_width, curr_width))

    # Open gripper
    des_width = rbq.max_width
    rbq.apply_commands(width=des_width)
    time.sleep(2)
    curr_width = rbq.get_sensors()
    print("RBQ:> Testing gripper Open: Desired:{}, Achieved:{}".format(des_width, curr_width))

    # Contineous control
    for i in range(int(time_to_go * hz)):
        des_width = rbq.max_width * ( 1 + np.cos(np.pi * i / (T * hz)) )/2
        rbq.apply_commands(width=des_width)
        time.sleep(1 / hz)

    # Drive gripper using keyboard
    if False:
        from vtils.keyboard import key_input as keyboard
        ky = keyboard.Key()
        sen = None
        print("Press 'q' to stop listening")
        while sen != 'q':
            sen = ky.get_sensor()
            if sen is not None:
                print(sen, end=", ", flush=True)
                if sen == 'up':
                    rbq.apply_commands(width=rbq.max_width)
                elif sen=='down':
                    rbq.apply_commands(width=rbq.min_width)
            time.sleep(.01)


    # close connection
    rbq.close()
    print("RBQ:> Demo Finished")
