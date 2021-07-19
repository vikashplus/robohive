import unittest
import gym
import mj_envs
import numpy as np


class TestEnvs(unittest.TestCase):

    def check_envs(self, module_name, env_names, lite=True):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
            print("Testing env: ", env_name)
            # test init
            env = gym.make(env_name)
            # test reset
            env.env.reset()
            # test obs vec
            # obs = env.env.get_obs()

            if not lite:
                # test obs dict
                obs_dict_sim = env.env.get_obs_dict(env.env.sim)
                obs_dict_sim_obsd = env.env.get_obs_dict(env.env.sim_obsd)

                # for key in obs_dict_sim.keys():
                # check if the env is partially observable
                for key in env.env.obs_keys:
                    if (obs_dict_sim[key] != obs_dict_sim_obsd[key]).any():
                        print("WARNING: {} environment is partially observable. This can happen if there aren't enough sensors defined in the robot config file or if you are using sensor noise. Please make sure this is a conscious choice".format(env_name))
                        print("\tobs_dict_sim[{}]: {}".format(key, obs_dict_sim[key]))
                        print("\tobs_dict_sim_obsd[{}]: {}\n".format(key, obs_dict_sim_obsd[key]))

                # test rewards
                rwd = env.env.get_reward_dict(obs_dict_sim)

                # test vector => dict upgrade
                # print(env.env.get_obs() - env.env.get_obs_vec())
                # assert (env.env.get_obs() == env.env.get_obs_vec()).all(), "check vectorized computations"

            # test env infos
            # infos = env.env.get_env_infos()

            # test step (everything together)
            observation, _reward, done, _info = env.env.step(np.zeros(env.env.sim.model.nu))

    # Franka Kitchen
    def test_frankakitchen(self):
        env_names = [
            # 'kitchen-v0',
            'kitchen-v2',
            'kitchen_micro_open-v2',
            'kitchen_rdoor_open-v2',
            'kitchen_ldoor_open-v2',
            'kitchen_sdoor_open-v2',
            'kitchen_light_on-v2',
            'kitchen_knob4_on-v2',
            'kitchen_knob3_on-v2',
            'kitchen_knob2_on-v2',
            'kitchen_knob1_on-v2',
            'kitchen-v3',
            'kitchen_micro_open-v3',
            'kitchen_rdoor_open-v3',
            'kitchen_ldoor_open-v3',
            'kitchen_sdoor_open-v3',
            'kitchen_light_on-v3',
            'kitchen_knob4_on-v3',
            'kitchen_knob3_on-v3',
            'kitchen_knob2_on-v3',
            'kitchen_knob1_on-v3'
            ]
        self.check_envs('Franka Kitchen', env_names, lite=False)

    # Arms
    def test_arms(self):
        env_names = [
            'FrankaReachFixed-v0',
            'FrankaReachRandom-v0',
            'FrankaPushFixed-v0',
            'FrankaPushRandom-v0',
            'FetchReachFixed-v0',
            'FetchReachRandom-v0']
        self.check_envs('Arms', env_names)

    # Functional Manipulation
    def test_fm(self):
        env_names = [
            # 'DManusReachFixed-v0',
            'FMReachFixed-v0'
            ]
        self.check_envs('Functional Manipulation', env_names)


if __name__ == '__main__':
    unittest.main()
