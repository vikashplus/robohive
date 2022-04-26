import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestKitchen(TestEnvs):
    def test_kitchen(self):
        env_names = [
        'kitchen-v3',
        'kitchen_close-v3',
        'kitchen_micro_open-v3',
        'kitchen_micro_close-v3',
        'kitchen_rdoor_open-v3',
        'kitchen_rdoor_close-v3',
        'kitchen_ldoor_open-v3',
        'kitchen_ldoor_close-v3',
        'kitchen_sdoor_open-v3',
        'kitchen_sdoor_close-v3',
        'kitchen_light_on-v3',
        'kitchen_light_off-v3',
        'kitchen_knob4_on-v3',
        'kitchen_knob4_off-v3',
        'kitchen_knob3_on-v3',
        'kitchen_knob3_off-v3',
        'kitchen_knob2_on-v3',
        'kitchen_knob2_off-v3',
        'kitchen_knob1_on-v3',
        'kitchen_knob1_off-v3',
        'franka_micro_open-v3','franka_micro_close-v3','franka_micro_random-v3',
        'franka_slide_open-v3','franka_slide_close-v3','franka_slide_random-v3',
        ]
        self.check_envs('Kitchen', env_names, lite=False)


if __name__ == '__main__':
    unittest.main()