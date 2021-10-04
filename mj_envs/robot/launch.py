import sys
import numpy as np
import redis
import torch
from threading import Lock, Thread
from time import time, sleep

from hardware_franka import FrankaArm, get_args, JointPDPolicy
from keyboard import getch

STATE_UPDATE_FREQ = 200
CMD_FREQ = 40
REDIS_STATE_KEY = 'robostate'
REDIS_CMD_KEY = 'robocmd'
CMD_SHAPE = 7
CMD_DTYPE = np.float64  # np.float64 for C++ double; np.float32 for float
CMD_DELTA_HIGH = np.array([0.05] * CMD_SHAPE)
CMD_DELTA_LOW = np.array([-0.05] * CMD_SHAPE)

def print_and_cr(msg): sys.stdout.write(msg + '\r\n')

class State(object):
    def __init__(self, franka, redis_store):
        self.franka = franka
        self.redis_store = redis_store
        self.quit = False
        self.mode = 'fix'
        self._mutex = Lock() # Not in use
        self.print_state = False

    def lock(self):
        self._mutex.acquire()

    def unlock(self):
        self._mutex.release()

def redis_send_states(redis_store, robostate):
    redis_store.set(REDIS_STATE_KEY, robostate.tobytes())

def redis_send_dummy_command(redis_store, robopos):
    redis_store.set(REDIS_CMD_KEY, robopos.tobytes())

def redis_receive_command(redis_store):
    return np.array(np.frombuffer(redis_store.get(REDIS_CMD_KEY), dtype=CMD_DTYPE).reshape(CMD_SHAPE))

def keyboard_proc(state):

    # Keyboard Interface
    res = getch()
    while res != 'q' and not state.quit: # Press q to quit
        with state._mutex:
            if res == 'p':
                state.print_state = True
            elif res == 'x':
                state.mode = 'fix'
            elif res == 'h':
                state.mode = 'home'
            elif res == 't':
                robostate = np.array(state.franka.get_sensors_offsets(), dtype=CMD_DTYPE)
                redis_send_dummy_command(state.redis_store, robostate) # TODO pick out pos
                state.mode = 'teleop'
            elif res == 'd':
                state.mode = 'debug'
            elif res == 's':
                state.mode = 'step'
        res = getch()

    state.quit = True


if __name__ == "__main__":

    args = get_args()

    # Initialize
    print_and_cr(f"Connecting to Franka robot at {args.server_ip} ...")
    franka = FrankaArm(name="Franka-Demo", ip_address=args.server_ip)
    franka.robot.go_home()
    my_default_policy = franka.default_policy(1.0, 1.0)
    franka.connect(policy=my_default_policy)
    print_and_cr(f"Connected to Franka arm")
    redis_store = redis.Redis()
    state = State(franka, redis_store)

    # Launch keyboard thread
    cmd_thread = Thread(target=keyboard_proc, name='Teleop Keyboard Thread', args=(state,))
    cmd_thread.start()

    ts = time()
    period = 1 / STATE_UPDATE_FREQ
    prev_joint = None

    while not state.quit:
        robostate = np.array(state.franka.get_sensors_offsets(), dtype=CMD_DTYPE)
        redis_send_states(state.redis_store, robostate)

        if state.print_state:
            print_and_cr(f"Current state {robostate}")
            state.print_state = False

        if state.franka.robot.get_previous_interval().end != -1:
            print_and_cr("<<<<<<< custom controller died")
            print_and_cr("Trying reconnect")
            state.franka.connect(policy=my_default_policy)

        if state.mode == 'teleop':
            joint_pos_desired = redis_receive_command(state.redis_store)
            #print_and_cr(f"Received joint command {joint_pos_desired}")
            #print_and_cr(f"Current robot state {robostate}")
            np.clip(joint_pos_desired, robostate+CMD_DELTA_LOW, robostate+CMD_DELTA_HIGH, out=joint_pos_desired)
            #print_and_cr(f"Clipped cmd's delta {joint_pos_desired - robostate}")
            state.franka.apply_commands_offsets(joint_pos_desired)

        elif state.mode == 'home':
            print_and_cr(f"Sending robot to home position. Takes about 4 sec. Do not move ..")
            state.franka.reset()
            print_and_cr(f"Franka is at home, sending our controller")
            state.franka.connect(policy=my_default_policy)
            state.mode = 'idle'

        elif state.mode == 'debug':
            joint_pos_desired = redis_receive_command(state.redis_store)
            print_and_cr(f"Current state {robostate}")
            print_and_cr(f"Received cmd {joint_pos_desired}")
            print_and_cr(f"Diff {joint_pos_desired - robostate}")
            state.mode = 'idle'

        elif state.mode == 'step':
            joint_pos_desired = np.array(robostate, dtype=CMD_DTYPE)
            joint_pos_desired[-1] += 0.03
            state.mode = 'idle'
            state.franka.apply_commands_offsets(joint_pos_desired)

            #ts = time()
        #ts += period
        #sleep(max(0, ts - time()))
        sleep( period )

    state.franka.close()

    print_and_cr("Close demo")
