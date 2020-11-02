from mj_envs.envs.arms.reach_base_v0 import ReachEnvFixed, ReachEnvRandom
import os

class FetchReachFixed(ReachEnvFixed):

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        ReachEnvRandom.__init__(self,
            model_path = '/fetch/assets/fetch_reach_v0.xml',
            config_path = curr_dir + '/assets/fetch_reach_v0.config',
            robot_site_name="grip",
            target_site_name="target")


class FetchReachRandom(ReachEnvRandom):

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        ReachEnvRandom.__init__(self,
            model_path = '/fetch/assets/fetch_reach_v0.xml',
            config_path = curr_dir + '/assets/fetch_reach_v0.config',
            robot_site_name="grip",
            target_site_name="target")