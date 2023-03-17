import unittest
from robohive.tests.test_envs import TestEnvs

class TestKitchen(TestEnvs):
    # def test_kitchen(self):
    #     env_names = [
    #     'kitchen-v3',
    #     'kitchen_rgb-v3',
    #     #substeps1
    #     'kitchen_micro_open-v3',
    #     'kitchen_micro_close-v3',
    #     'kitchen_rdoor_open-v3',
    #     'kitchen_rdoor_close-v3',
    #     'kitchen_ldoor_open-v3',
    #     'kitchen_ldoor_close-v3',
    #     'kitchen_sdoor_open-v3',
    #     'kitchen_sdoor_close-v3',
    #     'kitchen_light_on-v3',
    #     'kitchen_light_off-v3',
    #     'kitchen_knob4_on-v3',
    #     'kitchen_knob4_off-v3',
    #     'kitchen_knob3_on-v3',
    #     'kitchen_knob3_off-v3',
    #     'kitchen_knob2_on-v3',
    #     'kitchen_knob2_off-v3',
    #     'kitchen_knob1_on-v3',
    #     'kitchen_knob1_off-v3',
    #     'franka_micro_open-v3','franka_micro_close-v3','franka_micro_random-v3',
    #     'franka_slide_open-v3','franka_slide_close-v3','franka_slide_random-v3',
    #     #substeps2
    #     'kitchen_micro_slide_close-v3',''
    #     #substeps9
    #     'kitchen_openall-v3', 'kitchen_closeall-v3',
    #     ]
    #     self.check_envs('Kitchen', env_names, lite=False)

    def test_kitchen_v4(self):
        env_names = [
            'FK1_RelaxFixed-v4', 'FK1_MicroOpenFixed-v4', 'FK1_MicroCloseFixed-v4', 'FK1_RdoorOpenFixed-v4', 'FK1_RdoorCloseFixed-v4', 'FK1_LdoorOpenFixed-v4', 'FK1_LdoorCloseFixed-v4', 'FK1_SdoorOpenFixed-v4', 'FK1_SdoorCloseFixed-v4', 'FK1_LightOnFixed-v4', 'FK1_LightOffFixed-v4', 'FK1_Knob4OnFixed-v4', 'FK1_Knob4OffFixed-v4', 'FK1_Knob3OnFixed-v4', 'FK1_Knob3OffFixed-v4', 'FK1_Knob2OnFixed-v4', 'FK1_Knob2OffFixed-v4', 'FK1_Knob1OnFixed-v4', 'FK1_Knob1OffFixed-v4', 'FK1_Stove1KettleFixed-v4', 'FK1_Stove4KettleFixed-v4'
        ]
        self.check_envs('KitchenFixed-v4', env_names, lite=False)

        env_names = [
            'FK1_RelaxRandom-v4', 'FK1_MicroOpenRandom-v4', 'FK1_MicroCloseRandom-v4', 'FK1_RdoorOpenRandom-v4', 'FK1_RdoorCloseRandom-v4', 'FK1_LdoorOpenRandom-v4', 'FK1_LdoorCloseRandom-v4', 'FK1_SdoorOpenRandom-v4', 'FK1_SdoorCloseRandom-v4', 'FK1_LightOnRandom-v4', 'FK1_LightOffRandom-v4', 'FK1_Knob4OnRandom-v4', 'FK1_Knob4OffRandom-v4', 'FK1_Knob3OnRandom-v4', 'FK1_Knob3OffRandom-v4', 'FK1_Knob2OnRandom-v4', 'FK1_Knob2OffRandom-v4', 'FK1_Knob1OnRandom-v4', 'FK1_Knob1OffRandom-v4', 'FK1_Stove1KettleRandom-v4', 'FK1_Stove4KettleRandom-v4'
        ]
        self.check_envs('KitchenRandom-v4', env_names, lite=False)

        env_names = [
            'FK1_RelaxRandom_v2d-v4', 'FK1_MicroOpenRandom_v2d-v4', 'FK1_MicroCloseRandom_v2d-v4', 'FK1_RdoorOpenRandom_v2d-v4', 'FK1_RdoorCloseRandom_v2d-v4', 'FK1_LdoorOpenRandom_v2d-v4', 'FK1_LdoorCloseRandom_v2d-v4', 'FK1_SdoorOpenRandom_v2d-v4', 'FK1_SdoorCloseRandom_v2d-v4', 'FK1_LightOnRandom_v2d-v4', 'FK1_LightOffRandom_v2d-v4', 'FK1_Knob4OnRandom_v2d-v4', 'FK1_Knob4OffRandom_v2d-v4', 'FK1_Knob3OnRandom_v2d-v4', 'FK1_Knob3OffRandom_v2d-v4', 'FK1_Knob2OnRandom_v2d-v4', 'FK1_Knob2OffRandom_v2d-v4', 'FK1_Knob1OnRandom_v2d-v4', 'FK1_Knob1OffRandom_v2d-v4', 'FK1_Stove1KettleRandom_v2d-v4', 'FK1_Stove4KettleRandom_v2d-v4'
        ]
        self.check_envs('KitchenRandom_v2d-v4', env_names, lite=False)

if __name__ == '__main__':
    unittest.main()