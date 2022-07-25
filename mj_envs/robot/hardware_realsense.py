import numpy as np
# from hardware_base import hardwareBase
from mj_envs.robot.hardware_base import hardwareBase

import argparse
import a0
import logging
import copy
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import mj_envs.robot.serdes as serdes

sensor_msgs = serdes.get_capnp_msgs('sensor_msgs')

class RealSense(hardwareBase):
    def __init__(self, name, topic, data_types='rgb', **kwargs):
        self.data_types=data_types
        self.topic = topic
        self.last_image_pkt = None
        self.image_sub = None
        self.last_depth_pkt = None

    def connect(self):
        # sub to the self.topic
        self.image_sub = a0.Subscriber(self.topic, a0.ITER_NEWEST, self.callback)
        # self.depth_sub = a0.Subscriber(self.topic+'depth/image_raw', a0.ITER_NEWEST, self.callback_depth)

    def callback(self, pkt):
        # save the latest sensor reading into local var
        # logger.info("In callback")
        msg = sensor_msgs.Image.from_bytes(pkt.payload)
        dtype=np.dtype(msg.encoding)
        color_depth = int(msg.step/msg.width/dtype.itemsize)
        if len(msg.data) != msg.height*msg.width*color_depth*dtype.itemsize:
            logger.warning(f'{self.sub_topic}: short packet {len(msg.data)} < {msg.height*msg.width*color_depth*dtype.itemsize}! skip')
            return
        # print(msg.height, msg.width)

        if color_depth == 3:
            self.last_image_pkt = np.ndarray(shape=(msg.height, msg.width, color_depth), dtype=dtype, buffer=msg.data)
        else:
            self.last_depth_pkt = np.ndarray(shape=(msg.height, msg.width, color_depth), dtype=dtype, buffer=msg.data)

    def get_sensors(self):
        # get all data mentioned in data_types
        last_img = copy.deepcopy(self.last_image_pkt)
        last_depth = copy.deepcopy(self.last_depth_pkt)
        return {'rgb': last_img, 'd': last_depth}

    def apply_commands(self):
        return 0

    def close(self):
        return True

    def okay(self):
        return self.image_sub is not None

    def reset(self):
        return 0



# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="RealSense Client: Connects to realsense pub")

    parser.add_argument("-t", "--topic",
                        type=str,
                        help="topic name of the camera",
                        default="")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    rs = RealSense(name="test cam", topic=args.topic)
    rs.connect()
    assert rs.okay(), "Couldn't connect to the camera (topic: {})".format(args.topic)
    import cv2

    for i in range(50):
        img = rs.get_sensors()
        if img['rgb'] is not None:
            # print(img['rgb'])
            cv2.imshow("rgb", img['rgb'])
            cv2.waitKey(1)
        if img['d'] is not None:
            cv2.imshow("depth", img['d'])
            cv2.waitKey(1)
        time.sleep(0.1)