from hardware_base import hardwareBase
from dynamixel_py import dxl

class Dynamixels(hardwareBase):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        # initialize dynamixels
        ids = np.unique([device['sensor_ids'] + device['actuator_ids']]).tolist()
        device['robot'] = dxl(motor_id=ids, motor_type= device['interface']['motor_type'], devicename= device['interface']['name'])

    def connect(self):
        """Establish hardware connection"""
        device['robot'].open_port()

        # set actuator mode
        for actuator in device['actuator']:
            device['robot'].set_operation_mode(motor_id=[actuator['hdr_id']], mode=actuator['mode'])

        # engage motors
        device['robot'].engage_motor(motor_id=device['actuator_ids'], enable=True)

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