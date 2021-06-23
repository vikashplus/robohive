from mj_envs.envs.arms.relocate_box_base_v0 import RelocateBoxEnvFixed, RelocateBoxEnvRandom
import os

class FrankaRelocateBoxFixed(RelocateBoxEnvFixed):

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        RelocateBoxEnvFixed.__init__(self,
            model_path = '/franka/assets/franka_ycb_v0.xml',
            config_path = curr_dir + '/assets/franka_reach_v0.config',
            robot_site_name="end_effector",
            target_site_name="target",
            object_site_name="object",
            )