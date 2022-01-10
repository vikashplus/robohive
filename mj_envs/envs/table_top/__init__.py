import os
from gym.envs.registration import register

print("RS:> Registering Table Top Envs")
# Swing the door open
curr_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH=curr_dir+'/assets/reorient.xml'
register(
    id='reorient-fixed-v0',
    entry_point='mj_envs.envs.table_top:ReorientEnvV0',
    kwargs={
        "model_path": MODEL_PATH,
        #"config_path": CONFIG_PATH,
        #"obj_init": {"knob1_joint": -1.57},0 -0.2 0
        "goal": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # Sets the orientation of the body. (0,0,0) sets it to default
        #"obs_keys_wt": obs_keys_wt,
    },
    max_episode_steps=50,
    )
register(
    id='reorient-random-v0',
    entry_point='mj_envs.envs.table_top:ReorientEnvV0',
    kwargs={
        "model_path": MODEL_PATH,
        #"config_path": CONFIG_PATH,
        #"obj_init": {"knob1_joint": -1.57},0 -0.2 0
        "goal": [[0.0, 0.0], [0.0, 0.0], [0.0, 3.14]], # Sets the orientation of the body.
        #"obs_keys_wt": obs_keys_wt,
    },
    max_episode_steps=50,
)
from mj_envs.envs.table_top.reorient_v0 import ReorientEnvV0
