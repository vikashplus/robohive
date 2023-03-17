""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from darwin.darwin_robot.hardware_base import hardwareBase
import numpy as np
import socket
import argparse
import time
import threading
from collections import deque

_USE_UDP = True

class OptiTrack(hardwareBase):
    """
    OptiTrack Client: Connects to the server and receives streaming data
    INPUTS:
        ip:             name/ ip for connection (reciever for UDP and sender for TCP)
        port:           port for connection
        packet_size:    Bytes of data being exchanged
    """

    # Cached client that is shared for the application lifetime.
    _OPTI_CLIENT = None

    def __init__(self, ip: str, port:int=5000, packet_size:int=36, cache_maxsize:int=0):
        if self._OPTI_CLIENT is None:
            self.ip = ip
            self.port = port
            self.packet_size = packet_size
            self._sensor_cache_maxsize = cache_maxsize

            self.data_raw = None
            self.data_float = None
            if cache_maxsize > 1:
                self._sensor_cache = deque([], maxlen=self.sensor_cache_maxsize)
            else:
                self._sensor_cache = None
        else:
            print("Connection to OptiTrack exists")


    def connect(self):
        if self._OPTI_CLIENT is None:
            print("Connecting to: {}:{}...".format(self.ip, self.port))
            if _USE_UDP:
                self._OPTI_CLIENT = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # instantiate
                self._OPTI_CLIENT.bind((self.ip, self.port))
            else:
                self._OPTI_CLIENT = socket.socket()  # instanciate
                self._OPTI_CLIENT.connect((self.ip, self.port))  # connect to the server
            print("Connected to: {}:{}".format(self.ip, self.port))

            # wait for cache to initialize
            if self._sensor_cache:
                print("Initializing sensor cache. Polling {} samples".format(self.sensor_cache_maxsize))
                while(len(self._sensor_cache)< self.sensor_cache_maxsize):
                    time.sleep(.1)
                print("Finished polling. Sensor cache initialized")

            # start polling data
            self.read_thread = threading.Thread(target=self.poll_sensors, args=())
            self.read_thread.start()
            time.sleep(1)

        else:
            print("Connection to OptiTrack exists")


    def okay(self):
        if (self._OPTI_CLIENT is None):
            return False
        else:
            return True


    def close(self):
        self.poll = False
        self.read_thread.join() # wait for the thread to finish
        if self.okay():
            self._OPTI_CLIENT.close()  # close the connection
            self._OPTI_CLIENT = None
            print("Disconnected from: {}:{}".format(self.ip, self.port))
        return True


    def reset(self):
        self.close()
        self.connect()


    # [t, id, x, y, z, q0, q1, q2, q3, q4]
    def read_sensor(self):
        # receive response
        if _USE_UDP:
            data, addr = self._OPTI_CLIENT.recvfrom(self.packet_size)
        else:
            data = self._OPTI_CLIENT.recv(self.packet_size)

        if len(data)!=self.packet_size: # if not all data read. Read leftovers
            n_read = len(data)
            n_left = self.packet_size - n_read
            data += self._OPTI_CLIENT.recv(n_left)  # purge unread data
            print("Partial packet of length {} found at t={:3.3f}. Reading \
                additional {} bytes".format(n_read, self.data_float[0], n_left))

        # handle data
        self.data_raw = data
        self.data_float = np.frombuffer(data, dtype=np.float32).copy()

        # sensor_data isn't updated in place ==> it can be easily passed around and cached
        # dict is using the same buffer as the self.data_float to avoid mem allocation
        self.sensor_data = {'time': self.data_float[0:1], 'id':self.data_float[1:2],\
            'pos':self.data_float[2:5], 'quat':self.data_float[5:9]}

        if self._sensor_cache:
            self._sensor_cache.append(self.sensor_data)

    def poll_sensors(self):
        print("Start polling sensor data")
        self.poll = True
        while self.poll:
            self.read_sensor()
            # print(self)
        print("Finished polling sensor data")

    def __repr__(self):
        return "ET:{:03.3f}, id:{:2d}, x:{:+03.3f}, y:{:+03.3f}, z:{:+03.3f}"\
                .format(self.data_float[0], int(self.data_float[1]), \
                    self.data_float[2], self.data_float[3], self.data_float[4])

    # get latest sensor value (helpful when there is a single sensors)
    def get_sensors(self):
        # sensor_data isn't updated in place ==> it can be easily passed around and cached
        # repeated calls will return the same data_frame ==> no overhead for multiple queries to the same sensor reading
        return self.sensor_data

    # get sensor from cache (helpful when there are >1 sensors)
    def get_sensors_from_cache(self, index=-1):
        assert self._sensor_cache is not None, "Cache not initialized"
        assert (index>=0 and index<self._sensor_cache_maxsize) or \
                (index<0 and index>=-self._sensor_cache_maxsize), \
                "cache index out of bound. (cache size is %2d)"%self._sensor_cache_maxsize
        return self._sensor_cache[index]

    def apply_commands(self):
        raise NotImplementedError


# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="OptiTrack Client: Connects to \
        the server and fetches streaming data")

    parser.add_argument("-s", "--server_name",
                        type=str,
                        help="IP address or hostname of the streaming server",
                        default="xyz.cs.washington.edu") # 169.254.163.86
    parser.add_argument("-c", "--client_name",
                        type=str,
                        help="IP address or hostname of the recieving client",
                        default="zyx.cs.washington.edu") # 169.254.163.96
    parser.add_argument("-p", "--port",
                        type=int,
                        help="Port to use (> 1024)",
                        default=5000)
    parser.add_argument("-n", "--nbytes",
                        type=int,
                        help="Size of packets being exchanged",
                        default=36) # [t, id, x, y, z, q0, q1, q2, q3, q4]
    parser.add_argument("-v", "--verbose",
                        type=bool,
                        help="print data stream",
                        default=False)
    return parser.parse_args()



if __name__ == '__main__':

    # get args
    args = get_args()

    # Connect and receive streaming data
    if _USE_UDP:
        oc = OptiTrack(args.client_name, args.port, args.nbytes)
    else:
        oc = OptiTrack(args.server_name, args.port, args.nbytes)
    oc.connect()

    print(oc.get_sensor_from_cache(-5))

    print("Reading data.... (Press CTRL+C to exit)")
    while oc.okay():
        data = oc.get_sensors()
        if args.verbose:
            if data is None:
                print("Data: None")
            else:
                print("T:{:03.3f}, id:{:2d}, x:{:+03.3f}, y:{:+03.3f}, z:{:+03.3f}"\
                    .format(data[0], int(data[1]), data[2], data[3], data[4]))
            time.sleep(0.1)
    oc.close()