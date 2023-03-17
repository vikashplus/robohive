import enum
from typing import Union
import collections
import numpy as np
from robohive.utils.quat_math import euler2quat
import glob

# Time precision to use. Avoids rounding/resolution efforts during comparisions
_TIME_PRECISION = 4

# Reference structure
reference = collections.namedtuple('reference',
        ['time',        # int
         'robot',       # shape(N, n_robot_jnt) ==> robot trajectory
         'robot_vel',   # shape(N, n_robot_jnt) ==> robot velocity
         'object',      # shape(M, n_objects_jnt) ==> object trajectory
         'robot_init',  # shape(n_objects_jnt) ==> initial robot pose (can be different from robot(0,n_robot_jnt))
         'object_init'  # shape(n_objects_jnt) ==> initial object (can be different from object(0,n_object_jnt))
         ])


# Reference type
class ReferenceType(enum.Enum):
    """Reference types"""
    FIXED = 0
    RANDOM = 1
    TRACK = 2


# Reference motion
class ReferenceMotion():
    def __init__(self, reference:Union[str, dict],
                 motion_extrapolation:bool=False    # zero order hold if the motion is over
                 ):
        """
        Reference Type
            Fixed  :: N==1, M==1  :: input is <Fixed target dict>
            Random :: N==2, M==2  :: input is <Randomization range dict> :: ind0:low_limit, ind1:high_limit
            Track  :: N>2 or M>2  :: input as <Motion file to be tracked>
        """

        self.motion_extrapolation = motion_extrapolation
        # load reference
        if isinstance(reference, str):
            self.reference = self.load(reference)
            if 'robot_vel' not in self.reference.keys():
                self.reference['robot_vel'] = np.zeros(self.reference['robot'].shape)
        elif isinstance(reference, dict):
            self.reference = reference
        else:
            raise TypeError("Unknown reference type")

        # check reference for format
        self.check_format(self.reference)
        self.reference['time'] = np.around(self.reference['time'], _TIME_PRECISION) # round to help with comparisions
        robot_shape = self.reference['robot'].shape
        object_shape = self.reference['object'].shape
        self.robot_dim = robot_shape[1]
        self.object_dim = object_shape[1]

        # identify type
        if robot_shape[0] == 1 and object_shape[0] == 1:
            self.type = ReferenceType.FIXED
        elif robot_shape[0] == 2 and object_shape[0] == 2:
            self.type = ReferenceType.RANDOM
        elif robot_shape[0] > 2 or object_shape[0] > 2:
            self.type = ReferenceType.TRACK
        else:
            raise ValueError("Reference values not per specs")

        # identify length
        self.robot_horizon = robot_shape[0]
        self.object_horizon = object_shape[0]
        self.horizon = max(robot_shape[0], object_shape[0])

        # Add missing values
        if self.type == ReferenceType.RANDOM:
            # use mean postures from the randomization range
            if 'robot_init' not in self.reference.keys():
                self.reference['robot_init'] = np.mean(self.reference['robot'])
            if 'object_init' not in self.reference.keys():
                self.reference['object_init'] = np.mean(self.reference['object'][0])
        else:
            # use first timeset
            if 'robot_init' not in self.reference.keys():
                self.reference['robot_init'] = self.reference['robot'][0]
            if 'object_init' not in self.reference.keys():
                self.reference['object_init'] = self.reference['object'][0]

        # cache to help with finding index
        self.index_cache = 0

    def check_format(self, reference):
        """
        Check reference format
        """
        assert 'time' in reference.keys(), "Missing keys in reference"
        assert 'robot' in reference.keys(), "Missing keys in reference"
        assert 'robot_vel' in reference.keys(), "Missing keys in reference"
        assert 'object' in reference.keys(), "Missing keys in reference"
        assert reference['robot'].ndim == 2, "Check robot reference, must be shape(N, n_robot_jnt)"
        assert reference['object'].ndim == 2, "Check object reference, must be shape(N, n_object_jnt)"
        if 'robot_init' in reference.keys():
            assert reference['robot_init'].ndim == 1, "Check robot_init reference, must be shape(n_robot_jnt)"
            assert reference['robot_init'].shape[0] == reference['robot'].shape[1], "n_robot_jnt different between motion and init "
        if 'object_init' in reference.keys():
            assert reference['object_init'].ndim == 1, "Check object_init reference, must be shape(n_object_jnt)"
            assert reference['object_init'].shape[0] == reference['object'].shape[1], "n_object_jnt different between motion and init "


    def load(self, reference_path):
        """
        Load reference motion from the given path
        """
        return {k:v for k, v in np.load(reference_path).items()}


    def find_timeslot_in_reference(self, time):
        """
        Find the timeslot interval for the provided time in the reference motion
        INPUT:  Time: Specified time
        OUTPUT: Timeslot index tuple(ind_prev, ind_next)
                - where reference['time'][ind_prev] <= time <= reference['time'][ind_next]
                - ind_prev == ind_next if exact time is found in reference['time']
        """
        time = np.around(time, _TIME_PRECISION) # round to help with comparisions
        if self.motion_extrapolation and time>=self.reference['time'][-1]:
            return (self.horizon-1,self.horizon-1)
        else:
            assert time<=self.reference['time'][-1], f"Trying to access time (={time}) beyond max reference duration (={self.reference['time'][-1]}) "


        # search locally for index
        if time == self.reference['time'][self.index_cache]:
            # print(f"curr match: {time}")
            return (self.index_cache, self.index_cache)

        elif self.index_cache<(self.horizon-1):
            if time == self.reference['time'][self.index_cache+1]:
                # print(f"next match: {time}")
                self.index_cache += 1
                return (self.index_cache, self.index_cache)

            elif time > self.reference['time'][self.index_cache] and time < self.reference['time'][self.index_cache+1]:
                # print(f"interval match: {time}")
                return (self.index_cache, self.index_cache+1)
            else:
                self.index_cache = np.searchsorted(self.reference['time'], time, side="right") - 1
                print(f"No result using hueristic search. Attempting sort match: {time}")
                if time == self.reference['time'][self.index_cache]:
                    return (self.index_cache, self.index_cache)
                elif time > self.reference['time'][self.index_cache] and time < self.reference['time'][self.index_cache+1]:
                    return (self.index_cache, self.index_cache+1)
                else:
                    raise ValueError("We shouldn't be in this condition")

    def reset(self):
        self.index_cache = 0


    def get_reference(self, time):
        """
        Return the reference at the given time. Linearly interpolate if there is no exact match
        """
        if self.type == ReferenceType.FIXED:
            robot_ref = self.reference['robot'][0]
            robot_vel_ref = self.reference['robot_vel'][0]
            object_ref = self.reference['object'][0]
        elif self.type == ReferenceType.RANDOM:
            robot_ref     = self.np_random.uniform(low=self.reference['robot'][0,:], high=self.reference['robot'][1,:])
            robot_vel_ref = self.np_random.uniform(low=self.reference['robot_vel'][0,:], high=self.reference['robot_vel'][1,:])
            object_ref    = self.np_random.uniform(low=self.reference['object'][0,:], high=self.reference['object'][1,:])
        elif self.type == ReferenceType.TRACK:
            ind, ind_next = self.find_timeslot_in_reference(time=time)
            if ind == ind_next:
                # Exact frame[time] found for reference
                robot_ref     = self.reference['robot'][ind] if self.robot_horizon>1 else self.reference['robot'][0]
                robot_vel_ref = self.reference['robot_vel'][ind] if self.robot_horizon>1 else self.reference['robot_vel'][0]
                object_ref    = self.reference['object'][ind] if self.object_horizon>1 else self.reference['object'][0]
            else:
                # Linearly interpolate between frames to get references
                # ref[time] = blend(ref[ind] + (1-blend)*ref[ind_next])
                print(f"Direct frame reference not found at {time} sec. Attempting linear blend between two frames [{ind},{ind_next}]")
                blend = time - self.reference['time'][ind]/(self.reference['time'][ind_next]-self.reference['time'][ind])
                if self.robot_horizon>1:
                    robot_ref = (1.0-blend)**self.reference['robot'][ind]+blend*self.reference['robot'][ind_next]
                    robot_vel_ref = (1.0-blend)**self.reference['robot_vel'][ind]+blend*self.reference['robot_vel'][ind_next]
                else:
                    robot_ref = self.reference['robot'][0]
                    robot_vel_ref = self.reference['robot_vel'][0]
                if self.object_horizon>1:
                    object_ref = (1.0-blend)*self.reference['object'][ind]+blend*self.reference['object'][ind_next]
                else:
                    object_ref = self.reference['object'][0]

        return reference(time = time,
            robot = robot_ref,
            robot_vel = robot_vel_ref,
            object = object_ref,
            robot_init = self.reference['robot_init'],
            object_init = self.reference['object_init'],
            )
