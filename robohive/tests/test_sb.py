from stable_baselines3 import PPO
import robohive
from robohive.utils import gym

from robohive import robohive_arm_suite
for env_name in sorted(robohive_arm_suite):
    print(f"Training {env_name} ========================================")
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=2)
    break
