from enum import Flag
from polymetis import GripperInterface
from robohive.robot.hardware_base import hardwareBase

import numpy as np
import argparse
import time

class Spot(hardwareBase):
    #create init function
    def __init__(self, args):
        super().__init__(args)
        