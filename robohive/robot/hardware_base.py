""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

# Base robot class for other hardware devices to inheret from
import abc
import warnings

# Registry mapping robot_config's interface['type'] string -> hardwareBase subclass.
# Populated by @register_hardware(type_name) decorators on concrete hardware classes.
HARDWARE_REGISTRY = {}
SENSOR_POSTPROCESS = {}


def register_hardware(type_name, sensor_postprocess=None):
    """Registers a hardwareBase subclass (or a factory function returning one) against an
    interface['type'] string. Usable as a class decorator, or called directly with a
    factory function (e.g. one that tries a preferred implementation and falls back to
    an alternate one).

    sensor_postprocess(raw: dict) -> np.ndarray, optional: transforms the dict returned by
    get_sensors() into other formats.
    """
    def _decorator(cls_or_factory):
        if isinstance(cls_or_factory, type):
            assert issubclass(cls_or_factory, hardwareBase), \
                f"{cls_or_factory} must subclass hardwareBase to be registered"
        HARDWARE_REGISTRY[type_name] = cls_or_factory
        if sensor_postprocess is not None:
            SENSOR_POSTPROCESS[type_name] = sensor_postprocess
        return cls_or_factory
    return _decorator


class hardwareBase(abc.ABC):

    # add tests to all defined subclasses to ensure that get_sensors() returns a dict with a 'time' key
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'connect' in cls.__dict__:
            user_connect = cls.__dict__['connect']

            def connect_then_check(self, *args, **kw):
                result = user_connect(self, *args, **kw)
                try:
                    data = self.get_sensors()
                    if not (isinstance(data, dict) and 'time' in data):
                        warnings.warn(
                            f"{self.name}: get_sensors() should return a dict containing a 'time' key, got {type(data)}"
                        )
                except Exception as e:
                    warnings.warn(
                        f"{self.name}: could not verify get_sensors() 'time'-key contract after connect: {e}"
                    )
                return result

            cls.connect = connect_then_check

    def __init__(self, name, *args, **kwargs) -> None:
        self.name = name

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish hardware connection"""

    @abc.abstractmethod
    def okay(self) -> bool:
        """Check if hardware is healthy and return the status"""

    @abc.abstractmethod
    def recover(self) -> None:
        """Recover hardware from any error, connection loss, failure, etc """

    @abc.abstractmethod
    def close(self) -> bool:
        """Close hardware connection"""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset hardware to a known state. Used for resetting the hardware to a known state"""

    @abc.abstractmethod
    def get_sensors(self) -> dict:
        """Get hardware sensors — should return a dict containing a 'time' key"""

    @abc.abstractmethod
    def apply_commands(self) -> None:
        """Apply hardware commands"""

    def __del__(self) -> None:
        self.close()