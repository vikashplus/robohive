import unittest
from mj_envs.tests.test_envs import TestEnvs

class TestHMS(TestEnvs):
    # Hand Manipulation Suite
    def test_hand_manipulation_suite(self):
        env_names = [
        'pen-v1',
        'door-v1',
        'hammer-v1',
        'relocate-v1',
        # 'baoding-v1', 'baoding4th-v1', 'baoding8th-v1'
        ]
        self.check_envs('Hand Manipulation', env_names)

    def test_hand_manipulation_suite_visual(self):
        env_names = [
            'door_vr3m18-v1',
            'door_vr3m34-v1',
            'door_vr3m50-v1',
            'door_vrrl18-v1',
            'door_vrrl34-v1',
            'door_vrrl50-v1',
            'door_v2d-v1',
            'hammer_vr3m18-v1',
            'hammer_vr3m34-v1',
            'hammer_vr3m50-v1',
            'hammer_vrrl18-v1',
            'hammer_vrrl34-v1',
            'hammer_vrrl50-v1',
            'hammer_v2d-v1',
            'relocate_vr3m18-v1',
            'relocate_vr3m34-v1',
            'relocate_vr3m50-v1',
            'relocate_vrrl18-v1',
            'relocate_vrrl34-v1',
            'relocate_vrrl50-v1',
            'relocate_v2d-v1',
            'pen_vr3m18-v1',
            'pen_vr3m34-v1',
            'pen_vr3m50-v1',
            'pen_vrrl18-v1',
            'pen_vrrl34-v1',
            'pen_vrrl50-v1',
            'pen_v2d-v1'
            ]
        self.check_envs('Hand Manipulation(visual)', env_names)

if __name__ == '__main__':
    from mj_envs.utils.prompt_utils import set_prompt_verbosity, Prompt
    set_prompt_verbosity(verbose_mode=Prompt.WARN)
    unittest.main()
