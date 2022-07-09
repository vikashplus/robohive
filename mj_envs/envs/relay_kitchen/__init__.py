import os

from gym.envs.registration import register
from mj_envs.envs.relay_kitchen.kitchen_multitask_v1 import KitchenTasksV0
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaFixed as KitchenFranka
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaRandom
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaDemo
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaRandomDesk

print("RS:> Registering Kitchen Envs")


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = CURR_DIR + "/assets/franka_kitchen.xml"
CONFIG_PATH = CURR_DIR + "/assets/franka_kitchen.config"

# Kitchen
register(
    id="kitchen-v0",
    entry_point="mj_envs.envs.relay_kitchen:KitchenTasksV0",
    max_episode_steps=280,
)

# Kitchen
register(
    id="kitchen-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=280,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {},
    },
)

# Open Microwave door
register(
    id="kitchen_micro_open-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"microjoint": -1.25},
    },
)

# Open right hinge cabinet
register(
    id="kitchen_rdoor_open-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"rightdoorhinge": 1.57},
    },
)

# Open left hinge cabinet
register(
    id="kitchen_ldoor_open-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"leftdoorhinge": -1.25},
    },
)

# Open slide cabinet
register(
    id="kitchen_sdoor_open-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"slidedoor_joint": 0.44},
    },
)

# Lights on (left)
register(
    id="kitchen_light_on-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"lightswitch_joint": -0.7},
    },
)

# Knob4 on
register(
    id="kitchen_knob4_on-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"knob4_joint": -1.57},
    },
)

# Knob3 on
register(
    id="kitchen_knob3_on-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"knob3_joint": -1.57},
    },
)

# Knob2 on
register(
    id="kitchen_knob2_on-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"knob2_joint": -1.57},
    },
)

# Knob1 on
register(
    id="kitchen_knob1_on-v2",
    entry_point="mj_envs.envs.relay_kitchen:KitchenFranka",
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {"knob1_joint": -1.57},
    },
)


# ========================================================

# V3 environments
# In this version of the environment, the observations consist of the
# distance between end effector and all relevent objects in the scene

# from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import INTERACTION_SITES
obs_keys_wt = {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0, "end_effector": 1.0}
for site in KitchenFranka.INTERACTION_SITES:
    obs_keys_wt[site + "_err"] = 1.0

DEMO_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFrankaDemo"
RANDOM_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFrankaRandom"
FIXED_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFranka"
RANDOM_DESK_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFrankaRandomDesk"

# *** R3M Overide ***
# This changes the default behavior of the environment from a fixed kitchen to one
# where the kitchen randomly moves around, making the task visually more challenging.
FIXED_ENTRY_POINT = RANDOM_DESK_ENTRY_POINT

# Kitchen
register(
    id="kitchen-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=20,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {},
        "obj_init": {
            "knob1_joint": 0,
            "knob2_joint": 0,
            "knob3_joint": 0,
            "knob4_joint": 0,
            "lightswitch_joint": 0,
            "slidedoor_joint": 0,
            "microjoint": 0,
            "rightdoorhinge": 0,
            "leftdoorhinge": 0,
        },
        "obs_keys_wt": obs_keys_wt,
    },
)

# Kitchen
register(
    id="kitchen_close-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "goal": {},
        "obj_init": {
            "knob1_joint": -1.57,
            "knob2_joint": -1.57,
            "knob3_joint": -1.57,
            "knob4_joint": -1.57,
            "lightswitch_joint": -0.7,
            "slidedoor_joint": 0.44,
            "microjoint": -1.25,
            "rightdoorhinge": 1.57,
            "leftdoorhinge": -1.25,
        },
        "obs_keys_wt": obs_keys_wt,
    },
)

# Microwave door
register(
    id="kitchen_micro_open-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"microjoint": 0},
        "goal": {"microjoint": -1.25},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "microhandle_site",
    },
)
register(
    id="kitchen_micro_close-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"microjoint": -1.25},
        "goal": {"microjoint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "microhandle_site",
    },
)

# Right hinge cabinet
register(
    id="kitchen_rdoor_open-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"rightdoorhinge": 0},
        "goal": {"rightdoorhinge": 1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "rightdoor_site",
    },
)
register(
    id="kitchen_rdoor_close-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"rightdoorhinge": 1.57},
        "goal": {"rightdoorhinge": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "rightdoor_site",
    },
)

# Left hinge cabinet
register(
    id="kitchen_ldoor_open-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"leftdoorhinge": 0},
        "goal": {"leftdoorhinge": -1.25},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "leftdoor_site",
    },
)
register(
    id="kitchen_ldoor_close-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"leftdoorhinge": -1.25},
        "goal": {"leftdoorhinge": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "leftdoor_site",
    },
)

# Slide cabinet
register(
    id="kitchen_sdoor_open-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"slidedoor_joint": 0},
        "goal": {"slidedoor_joint": 0.44},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "slide_site",
    },
)
register(
    id="kitchen_sdoor_close-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"slidedoor_joint": 0.44},
        "goal": {"slidedoor_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "slide_site",
    },
)

# Lights
register(
    id="kitchen_light_on-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"lightswitch_joint": 0},
        "goal": {"lightswitch_joint": -0.7},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "light_site",
    },
)
register(
    id="kitchen_light_off-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"lightswitch_joint": -0.7},
        "goal": {"lightswitch_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "light_site",
    },
)

# Knob4
register(
    id="kitchen_knob4_on-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob4_joint": 0},
        "goal": {"knob4_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob4_site",
    },
)
register(
    id="kitchen_knob4_off-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob4_joint": -1.57},
        "goal": {"knob4_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob4_site",
    },
)

# Knob3
register(
    id="kitchen_knob3_on-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob3_joint": 0},
        "goal": {"knob3_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob3_site",
    },
)
register(
    id="kitchen_knob3_off-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob3_joint": -1.57},
        "goal": {"knob3_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob3_site",
    },
)

# Knob2
register(
    id="kitchen_knob2_on-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob2_joint": 0},
        "goal": {"knob2_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob2_site",
    },
)
register(
    id="kitchen_knob2_off-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob2_joint": -1.57},
        "goal": {"knob2_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob2_site",
    },
)

# Knob1
register(
    id="kitchen_knob1_on-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob1_joint": 0},
        "goal": {"knob1_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob1_site",
    },
)
register(
    id="kitchen_knob1_off-v3",
    entry_point=FIXED_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob1_joint": -1.57},
        "goal": {"knob1_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob1_site",
    },
)
