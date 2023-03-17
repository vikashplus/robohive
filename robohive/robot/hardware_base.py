""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

# Base robot class for other hardware devices to inheret from
import abc

class hardwareBase(abc.ABC):
    def __init__(self, name, *args, **kwargs):
        self.name = name

    @abc.abstractmethod
    def connect(self):
        """Establish hardware connection"""

    @abc.abstractmethod
    def okay(self):
        """Return hardware health"""

    @abc.abstractmethod
    def close(self):
        """Close hardware connection"""

    @abc.abstractmethod
    def reset(self):
        """Reset hardware"""

    @abc.abstractmethod
    def get_sensors(self):
        """Get hardware sensors"""

    @abc.abstractmethod
    def apply_commands(self):
        """Apply hardware commands"""

    def __del__(self):
        self.close()