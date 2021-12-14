import time
from time import sleep
import threading

from polymetis import GripperInterface
import polymetis_pb2

def get_gripper_state(gripper_interface, lock, cache):
    while True:
        with lock:
            cache['width'] = gripper_interface.get_state().width
            #print(f"<- {cache['width']}")
        sleep(0.01)

def send_gripper_cmd(grpc_connection, lock, cache):
    while True:
        with lock:
            width, speed, force = cache['cmd']
            #print(f"-> {width}")
        grpc_connection.Goto(polymetis_pb2.GripperCommand(width=width, speed=speed, force=force))
        sleep(0.01)

class Gripper():
    cache = {
        'width': 0.,
        'cmd': (0., 0.2, 0.1),
    }

    def __init__(self, ip_address):
        self.gripper = GripperInterface(ip_address=ip_address)
        init_gripper_state = self.gripper.get_state()
        self.gripper_max_width = init_gripper_state.max_width

        # Get State Cache
        self.gripper_state_lock = threading.Lock()
        self.gripper_state_thread = threading.Thread(
            target=get_gripper_state,
            args=(self.gripper, self.gripper_state_lock, Gripper.cache, ),
            daemon=True).start()

        # Command Cache
        self.gripper_cmd_lock = threading.Lock()
        self.gripper_cmd_thread = threading.Thread(
            target=send_gripper_cmd,
            args=(self.gripper.grpc_connection, self.gripper_cmd_lock, Gripper.cache,),
            daemon=True).start()

    def get(self):
        with self.gripper_state_lock:
            return Gripper.cache['width']

    def goto(self, width : float, speed : float, force : float):
        # Option 3. Update the cached command to send
        with self.gripper_cmd_lock:
            Gripper.cache['cmd'] = (width, speed, force)

        # Option 2. Open a thread for every cmd
        # threading.Thread(
        #    target=self.gripper.grpc_connection.Goto,
        #    args=(polymetis_pb2.GripperCommand(width=width, speed=speed, force=force),),
        #    daemon=True).start()

        # Option 1. Naive send cmd. Blocking
        # self.gripper.grpc_connection.Goto(polymetis_pb2.GripperCommand(width=width, speed=speed, force=force))

if __name__ == "__main__":
    gripper = Gripper("172.16.0.1")
    print(f"Get width: {gripper.get()}")
    print(f"Closing gripper ...")
    gripper.goto(width=0., speed=0.1, force=0.1)
    sleep(1)
    print(f"Get width: {gripper.get()}")
    print(f"Opening gripper ...")
    gripper.goto(width=gripper.gripper_max_width, speed=0.1, force=0.1)
    sleep(1)
    print(f"Get width: {gripper.get()}")
    sleep(1)
    print(f"Trying to oscillate between open / close")
    import numpy as np
    oscilation_w = gripper.gripper_max_width * \
        np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 0.5, 0.1])
    for w in oscilation_w:
        gripper.goto(width=w, speed=0.1, force=0.1)
        sleep(0.3)
        print(f"Get width: {gripper.get()}")
    sleep(1)
    print("Finished")
