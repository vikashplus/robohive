""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import time

import numpy as np
from dynamixel_py import dxl

from .hardware_base import hardwareBase, register_hardware


@register_hardware('dynamixel')
class Dynamixels(hardwareBase):
    """
    Adapter around dynamixel_py's dxl client. A single dynamixel bus is shared across all
    of a device's sensors and actuators, and needs per-motor mode ('Position'/'PWM') at
    connect time — that per-actuator detail lives in the robot_config device dict (not in
    `interface`), so this class opts in to receiving it (see Robot.hardware_init()) rather
    than requiring config authors to duplicate motor IDs/modes into `interface` too. This
    is a deliberately scoped exception for dynamixel-bus hardware (which genuinely needs
    per-motor mode bookkeeping); other hardware classes stay interface-only.

    interface config keys:
        motor_type : dynamixel motor series (e.g. 'X')
        port       : serial port (e.g. '/dev/ttyUSB0')
    """

    def __init__(self, name, motor_type, port, device, **kwargs):
        self.name = name
        self.motor_type = motor_type
        self.port = port
        self.device = device
        self.motor_ids = np.unique(device['sensor_ids'] + device['actuator_ids']).tolist()
        self.dxls = None

    def connect(self) -> bool:
        """Establish hardware connection"""
        self.dxls = dxl(motor_id=self.motor_ids, motor_type=self.motor_type, devicename=self.port)
        self.dxls.open_port()

        # set actuator mode
        for actuator in self.device['actuator']:
            self.dxls.set_operation_mode(motor_id=[actuator['hdr_adr']], mode=actuator['mode'])

        # engage motors
        self.dxls.engage_motor(motor_id=self.device['actuator_ids'], enable=True)
        return True

    def okay(self) -> bool:
        """Return hardware health"""
        return self.dxls is not None

    def recover(self) -> None:
        """Recover hardware from any error, connection loss, failure, etc"""
        self.close()
        self.connect()

    def close(self) -> bool:
        """Close hardware connection"""
        if self.dxls:
            status = self.dxls.close(self.motor_ids)
            if status:
                self.dxls = None
            return status
        return True

    def reset(self, hw_q=None) -> None:
        """Reset hardware to a known state"""
        if hw_q is not None:
            self.apply_commands(hw_q)

    def get_sensors(self) -> dict:
        """Get hardware sensors. 'pos'/'vel' are positionally ordered to match
        device['sensor_ids'] (i.e. device['sensor'] declaration order)."""
        return {
            'time': time.time(),
            'pos': self.dxls.get_pos(self.device['sensor_ids']),
            'vel': self.dxls.get_vel(self.device['sensor_ids']),
        }

    def apply_commands(self, hw_q) -> None:
        """Apply hardware commands. hw_q is positionally ordered to match device['actuator']."""
        pos_ids, pos_ctrl, pwm_ids, pwm_ctrl = [], [], [], []
        for i, actuator in enumerate(self.device['actuator']):
            val = hw_q[i]
            mode = actuator['mode']
            if mode == 'Position':
                pos_ids.append(actuator['hdr_adr'])
                pos_ctrl.append(val)
            elif mode == 'PWM':
                pwm_ids.append(actuator['hdr_adr'])
                pwm_ctrl.append(val)
            else:
                raise NotImplementedError(f"Actuator mode {mode} not found")
        if pos_ids:
            self.dxls.set_des_pos(pos_ids, pos_ctrl)
        if pwm_ids:
            self.dxls.set_des_pwm(pwm_ids, pwm_ctrl)

    def __del__(self):
        self.close()
