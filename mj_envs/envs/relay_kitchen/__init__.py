from gym.envs.registration import register
import numpy as np

# Kitchen
register(
    id='kitchen-v0',
    entry_point='mj_envs.envs.relay_kitchen:KitchenTasksV0',
    max_episode_steps=280,
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v1 import KitchenTasksV0

# Kitchen
register(
    id='kitchen-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=280,
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Open Microwave door
register(
    id='kitchen_micro_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0, 0, 0, -2.19]),
                'interact_site': "microhandle_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Open right hinge cabinet
register(
    id='kitchen_rdoor_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0, 0, 1.57, 0]),
                'interact_site': "hinge_site2"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Open left hinge cabinet
register(
    id='kitchen_ldoor_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0, -1.57, 0, 0]),
                'interact_site': "hinge_site1"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Open slide cabinet
register(
    id='kitchen_sdoor_open-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0.44, 0, 0, 0]),
                'interact_site': "slide_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Lights on (left)
register(
    id='kitchen_light_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, -.7, 0, 0, 0, 0]),
                'interact_site': "light_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Knob4 on
register(
    id='kitchen_knob4_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, -1.57, 0, 0, 0, 0, 0]),
                'interact_site': "knob4_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Knob3 on
register(
    id='kitchen_knob3_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, -1.57, 0, 0, 0, 0, 0, 0]),
                'interact_site': "knob3_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Knob2 on
register(
    id='kitchen_knob2_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, -1.57, 0, 0, 0, 0, 0, 0]),
                'interact_site': "knob2_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed

# Knob1 on
register(
    id='kitchen_knob1_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, -1.57, 0, 0, 0, 0, 0, 0]),
                'interact_site': "knob1_site"
            }
)
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed