import numpy as np
# from hardware_base import hardwareBase
from robohive.robot.hardware_base import hardwareBase

import argparse
import a0
import logging
import copy
import time
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import robohive.robot.serdes as serdes

sensor_msgs = serdes.get_capnp_msgs('sensor_msgs')

class RealSense(hardwareBase):
    def __init__(self, name, rgb_topic=None, d_topic=None, **kwargs):
        assert rgb_topic or d_topic, "Atleast one of the topics is needed"
        self.rgb_topic = rgb_topic
        self.d_topic = d_topic
        self.last_image_pkt = None
        self.rgb_sub = None
        self.last_depth_pkt = None
        self.most_recent_pkt_ts = None
        self.timeout = 1 # in seconds

    def connect(self):
        # sub to the topics
        if self.rgb_topic:
            self.rgb_sub = a0.Subscriber(self.rgb_topic, a0.ITER_NEWEST, self.callback)
        if self.d_topic:
            self.d_sub = a0.Subscriber(self.d_topic, a0.ITER_NEWEST, self.callback)
        while not self.okay():
            time.sleep(0.1)

    def callback(self, pkt):
        # save the latest sensor reading into local var
        # logger.info("In callback")
        with sensor_msgs.Image.from_bytes(pkt.payload) as msg:
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
        # get all data from all topics
        last_img = copy.deepcopy(self.last_image_pkt)
        last_depth = copy.deepcopy(self.last_depth_pkt)
        return {'time':self.most_recent_pkt_ts, 'rgb': last_img, 'd': last_depth}

    def apply_commands(self):
        return 0

    def close(self):
        return True

    def okay(self):
        if self.rgb_topic and (self.rgb_sub is None):
            print("WARNING: No subscriber found for topic: ", self.rgb_topic)
            return False

        if self.d_topic and (self.d_sub is None):
            print("WARNING: No subscriber found for topic: ", self.d_topic)
            return False

        if self.most_recent_pkt_ts is None:
            print("WARNING: No packets received yet from the realsense subscibers: ", self.rgb_topic, self.d_topic)
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
    parser = argparse.ArgumentParser(description="RealSense Client: Connects to realsense pub.\n"
    "\nExample: python robot/hardware_realsense.py -r realsense_815412070341/color/image_raw -d realsense_815412070341/depth_uncolored/image_raw")

    parser.add_argument("-r", "--rgb_topic",
                        type=str,
                        help="rgb_topic name of the camera",
                        default="")
    parser.add_argument("-d", "--d_topic",
                        type=str,
                        help="rgb_topic name of the camera",
                        default="")
    parser.add_argument("-v", "--view",
                        type=None,
                        help="Choice: CV2",
                        )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    print(args)
    rs = RealSense(name="test cam", rgb_topic=args.rgb_topic, d_topic=args.d_topic)
    rs.connect()
    assert rs.okay(), "Couldn't connect to the camera (rgb_topic: {})".format(args.rgb_topic)

    if args.view=='CV2':
        import cv2

    for i in range(50):
        img = rs.get_sensors()
        if img['rgb'] is not None:
            print("Received image{} of size:".format(i), img['rgb'].shape, flush=True)
            if args.view=='CV2':
                cv2.imshow("rgb", img['rgb'])
                cv2.waitKey(1)

        if img['d'] is not None:
            if args.view=='CV2':
                cv2.imshow("depth", img['d'])
                cv2.waitKey(1)
            print("Received depth{} of size:".format(i), img['d'].shape, flush=True)

        if img['rgb'] is None and img['d'] is None:
            print(img)

        time.sleep(0.1)
