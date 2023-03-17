""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from .hardware_base import hardwareBase
from dynamixel_py import dxl
import numpy as np

class Dynamixels(hardwareBase):
    def __init__(self, name, motor_ids, motor_type, devicename, **kwargs):
        self.name = name
        # initialize dynamixels
        self.dxls = dxl(motor_id=motor_ids, motor_type=motor_type, devicename=devicename)
        self.motor_ids = motor_ids

    def connect(self):
        """Establish hardware connection"""
        self.dxls.open_port()

        # set actuator mode
        for actuator in device['actuator']:
            self.dxls.set_operation_mode(motor_id=[actuator['hdr_id']], mode=actuator['mode'])

        # engage motors
        self.dxls.engage_motor(motor_id=self.dxls, enable=True)

    def okay(self):
        """Return hardware health"""

    def close(self):
        """Close hardware connection"""

    def reset(self):
        """Reset hardware"""

    def get_sensors(self):
        """Get hardware sensors"""

    def apply_commands(self):
        """Apply hardware commands"""

    def __del__(self):
        self.close()