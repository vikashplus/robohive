from mj_envs.envs.arms.reach_base_v0 import ReachEnvFixed, ReachEnvRandom
import os

class FrankaReachFixed(ReachEnvFixed):

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        ReachEnvFixed.__init__(self,
            model_path = '/franka/assets/franka_reach_v0.xml',
            config_path = curr_dir + '/assets/franka_reach_v0.config',
            robot_site_name="end_effector",
            target_site_name="target")


class FrankaReachRandom(ReachEnvRandom):

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        ReachEnvRandom.__init__(self,
            model_path = '/franka/assets/franka_reach_v0.xml',
            config_path = curr_dir + '/assets/franka_reach_v0.config',
            robot_site_name="end_effector",
            target_site_name="target")