from mj_envs.envs.relay_kitchen.kitchen_multitask_v1 import KitchenTasksV0
# from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaFixed as KitchenFranka
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaRandom as KitchenFranka
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFrankaDemo
from gym.envs.registration import register
# import numpy as np

# Kitchen
register(
    id='kitchen-v0',
    entry_point='mj_envs.envs.relay_kitchen:KitchenTasksV0',
    max_episode_steps=280,
)

# Kitchen
register(
    id='kitchen-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=280,
        kwargs={
                'goal': {},
           }
)

# Open Microwave door
register(
    id='kitchen_micro_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'microjoint': -1.25},
            }
)

# Open right hinge cabinet
register(
    id='kitchen_rdoor_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'rightdoorhinge':1.57},
            }
)

# Open left hinge cabinet
register(
    id='kitchen_ldoor_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'leftdoorhinge':-1.25},
            }
)

# Open slide cabinet
register(
    id='kitchen_sdoor_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'slidedoor_joint':0.44},
            }
)

# Lights on (left)
register(
    id='kitchen_light_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'lightswitch_joint':-.7},
            }
)

# Knob4 on
register(
    id='kitchen_knob4_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'knob4_joint':-1.57},
            }
)

# Knob3 on
register(
    id='kitchen_knob3_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'knob3_joint':-1.57},
            }
)

# Knob2 on
register(
    id='kitchen_knob2_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'knob2_joint':-1.57},
            }
)

# Knob1 on
register(
    id='kitchen_knob1_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFranka',
    max_episode_steps=50,
    kwargs={
                'goal': {'knob1_joint':-1.57},
            }
)


# ========================================================

# V3 environments
# In this version of the environment, the observations consist of the
# distance between end effector and all relevent objects in the scene

# from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import INTERACTION_SITES
obs_keys_wt = {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0, "end_effector": 1.0}
for site in KitchenFranka.INTERACTION_SITES:
    obs_keys_wt[site+'_err'] = 1.0

DEMO_ENTRY_POINT = 'mj_envs.envs.relay_kitchen:KitchenFrankaDemo'
RANDOM_ENTRY_POINT = 'mj_envs.envs.relay_kitchen:KitchenFranka'

# Kitchen
register(
    id='kitchen-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=280,
    kwargs={
                'goal': {},
                'obs_keys_wt': obs_keys_wt,
           }
)

# Open Microwave door
register(
    id='kitchen_micro_open-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'microjoint': -1.25},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'microhandle_site',
            }
)

# Open right hinge cabinet
register(
    id='kitchen_rdoor_open-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'rightdoorhinge':1.57},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'rightdoor_site',
            }
)

# Open left hinge cabinet
register(
    id='kitchen_ldoor_open-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'leftdoorhinge':-1.25},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'leftdoor_site',
            }
)

# Open slide cabinet
register(
    id='kitchen_sdoor_open-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'slidedoor_joint':0.44},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'slide_site',
            }
)

# Lights on (left)
register(
    id='kitchen_light_on-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'lightswitch_joint':-.7},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'light_site',
            }
)

# Knob4 on
register(
    id='kitchen_knob4_on-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFrankaDemo',
    max_episode_steps=50,
    kwargs={
                'goal': {'knob4_joint':-1.57},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'knob4_site',
            }
)

# Knob3 on
register(
    id='kitchen_knob3_on-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'knob3_joint':-1.57},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'knob3_site',
            }
)

# Knob2 on
register(
    id='kitchen_knob2_on-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'knob2_joint':-1.57},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'knob2_site',
            }
)

# Knob1 on
register(
    id='kitchen_knob1_on-v3',
    entry_point=DEMO_ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
                'goal': {'knob1_joint':-1.57},
                'obs_keys_wt': obs_keys_wt,
                'interact_site': 'knob1_site',
            }
)