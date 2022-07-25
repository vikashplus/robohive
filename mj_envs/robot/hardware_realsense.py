import numpy as np
# from hardware_base import hardwareBase
from mj_envs.robot.hardware_base import hardwareBase

import argparse
import a0
import logging
import copy
import time
import datetime

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
        self.most_recent_pkt_ts = None
        self.timeout = 1 # in seconds

    def connect(self):
        # sub to the self.topic
        self.image_sub = a0.Subscriber(self.topic, a0.ITER_NEWEST, self.callback)
        # self.depth_sub = a0.Subscriber(self.topic+'depth/image_raw', a0.ITER_NEWEST, self.callback_depth)
        while not self.okay():
            time.sleep(0.1)

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

        # Update the timestamp
        timestamp_str = dict(pkt.headers)["a0_time_wall"]
        # Python datetime cannot handle nano-seconds
        timestamp_str_wo_nano = timestamp_str[:23] + timestamp_str[29:]
        self.most_recent_pkt_ts = datetime.datetime.fromisoformat(timestamp_str_wo_nano)


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
        if self.image_sub is None or self.most_recent_pkt_ts is None:
            print("WARNING: No packets received yet from the image subsciber: ", self.topic)
            return False
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            okay_age_threshold = datetime.timedelta(seconds=self.timeout)
            time_delay = now - self.most_recent_pkt_ts
            if time_delay>okay_age_threshold:
                print("Significant signal delay: ", time_delay)
                return False

        return True

    def reset(self):
        return 0



# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="RealSense Client: Connects to realsense pub")

    parser.add_argument("-t", "--topic",
                        type=str,
                        help="topic name of the camera",
                        default="")
    parser.add_argument("-d", "--display",
                        type=None,
                        help="Choice: CV2",
                        )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    print(args)
    rs = RealSense(name="test cam", topic=args.topic)
    rs.connect()
    assert rs.okay(), "Couldn't connect to the camera (topic: {})".format(args.topic)

    if args.display=='CV2':
        import cv2

    for i in range(50):
        img = rs.get_sensors()
        if img['rgb'] is not None:
            print("Received image{} of size:".format(i), img['rgb'].shape, flush=True)
            if args.display=='CV2':
                cv2.imshow("rgb", img['rgb'])
                cv2.waitKey(1)
        elif img['d'] is not None:
            cv2.imshow("depth", img['d'])
            cv2.waitKey(1)
        else:
            print(img)
        time.sleep(0.1)