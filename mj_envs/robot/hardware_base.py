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