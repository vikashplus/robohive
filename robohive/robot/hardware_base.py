""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

# Base robot class for other hardware devices to inheret from
import abc
import warnings


class hardwareBase(abc.ABC):
    def __init__(self, name, *args, **kwargs) -> None:
        self.name = name

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish hardware connection"""

    @abc.abstractmethod
    def okay(self) -> bool:
        """Return hardware health"""

    @abc.abstractmethod
    def close(self) -> bool:
        """Close hardware connection"""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset hardware"""

    @abc.abstractmethod
    def _get_sensors(self) -> dict:
        """Get hardware sensors — returned dict must include a 'time' key"""

    def get_sensors(self) -> dict:
        """Get hardware sensors, enforcing 'time' key contract"""
        data = self._get_sensors()
        if not (isinstance(data, dict) and 'time' in data):
            warnings.warn(
                f"{self.name}: get_sensors() should return a dict containing a 'time' key, got {type(data)}. "
                "Please add 'time' details to your sensor data to suppress this warning.")
        return data

    @abc.abstractmethod
    def apply_commands(self) -> None:
        """Apply hardware commands"""

    def __del__(self) -> None:
        self.close()