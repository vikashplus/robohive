import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict


class RealsenseAPI:
    """Wrapper that implements boilerplate code for RealSense cameras"""

    def __init__(self, device_id=None, height=480, width=640, fps=30, warm_start=30, type=None,):
        self.height = height
        self.width = width
        self.fps = fps
        self.device_id = device_id
        self.warm_start=warm_start
        self.pipe = None


    def connect(self):
        try:
            self.pipe= rs.pipeline()
            config = rs.config()

            config.enable_device(self.device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            self.profile = self.pipe.start(config)
            self.align = rs.align(rs.stream.color)

            for _ in range(self.warm_start):
                self._get_frames()

        except Exception as e:
            print(e)
            device_ls = []
            for c in rs.context().query_devices():
                device_ls.append(c.get_info(rs.camera_info.serial_number))
            raise RuntimeError(f'please init with one of the device ids {device_ls}')

    def get_intrinsics_dict(self):
        stream = self.profile.get_streams()[1]
        intrinsics = stream.as_video_stream_profile().get_intrinsics()
        param_dict = dict([(p, getattr(intrinsics, p)) for p in dir(intrinsics) if not p.startswith('__')])
        param_dict['model'] = param_dict['model'].name
        return param_dict

    def _get_frames(self):
        if self.pipe is None:
            raise RuntimeError('please connect first')
        frameset = self.pipe.wait_for_frames()
        return self.align.process(frameset)

    def get_rgbd(self):
        """
        returns np.ndarray [h, w, 4] where first 3 channels are RGB[0-255] and last channels is depth in millimeters
        """
        frameset = self._get_frames()

        rgbd = np.empty([self.height, self.width, 4], dtype=np.uint16)

        color_frame = frameset.get_color_frame()
        rgbd[:, :, :3] = np.asanyarray(color_frame.get_data())

        depth_frame = frameset.get_depth_frame()
        rgbd[:, :, 3] = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units() * 1000  # in millimeters

        return rgbd

    def get_sensors(self):
        # get all data from all topics
        rgbd = self.get_rgbd()
        return {'time':0, 'rgb': rgbd[:, :, :3], 'd': rgbd[:, :, 3]}

    def okay(self):
        return True

    def apply_commands(self):
        return 0

    def close(self):
        return True

    def reset(self):
        return 0

if __name__ == "__main__":
    cam = RealsenseAPI()
    cam.connect()
    print(cam.get_intrinsics_dict())

    for i in range(100):
        rgbd = cam.get_rgbd()
        print(i,rgbd.shape)