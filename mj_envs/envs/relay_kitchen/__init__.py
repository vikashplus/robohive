from mj_envs.envs.relay_kitchen.kitchen_multitask_v1 import KitchenTasksV0
from mj_envs.envs.relay_kitchen.kitchen_multitask_v2 import KitchenFetchFixed
from gym.envs.registration import register
import numpy as np

# Kitchen
register(
    id='kitchen-v0',
    entry_point='mj_envs.envs.relay_kitchen:KitchenTasksV0',
    max_episode_steps=280,
)

# Kitchen
register(
    id='kitchen-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=280,
)

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

# Knob2 on
register(
    id='kitchen_knob2_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, -1.57, 0, 0, 0, 0, 0, 0, 0]),
                'interact_site': "knob2_site"
            }
)

# Knob1 on
register(
    id='kitchen_knob1_on-v2',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([-1.57, 0, 0, 0, 0, 0, 0, 0, 0]),
                'interact_site': "knob1_site"
            }
)


# ========================================================

# V3 environments
# This version contains fewer observation keys
# In particular, observations only consist of robot pose, object pose, and goals

# Kitchen
register(
    id='kitchen-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=280,
    kwargs={
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
           }
)

# Open Microwave door
register(
    id='kitchen_micro_open-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0, 0, 0, -2.19]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "microhandle_site"
            }
)

# Open right hinge cabinet
register(
    id='kitchen_rdoor_open-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0, 0, 1.57, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "hinge_site2"
            }
)

# Open left hinge cabinet
register(
    id='kitchen_ldoor_open-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0, -1.57, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "hinge_site1"
            }
)

# Open slide cabinet
register(
    id='kitchen_sdoor_open-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, 0, 0.44, 0, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "slide_site"
            }
)

# Lights on (left)
register(
    id='kitchen_light_on-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, 0, -.7, 0, 0, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "light_site"
            }
)

# Knob4 on
register(
    id='kitchen_knob4_on-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, 0, -1.57, 0, 0, 0, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "knob4_site"
            }
)

# Knob3 on
register(
    id='kitchen_knob3_on-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, 0, -1.57, 0, 0, 0, 0, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "knob3_site"
            }
)

# Knob2 on
register(
    id='kitchen_knob2_on-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([0, -1.57, 0, 0, 0, 0, 0, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "knob2_site"
            }
)

# Knob1 on
register(
    id='kitchen_knob1_on-v3',
    entry_point='mj_envs.envs.relay_kitchen:KitchenFetchFixed',
    max_episode_steps=50,
    kwargs={
                'goal': np.array([-1.57, 0, 0, 0, 0, 0, 0, 0, 0]),
                'obs_keys_wt': {"hand_jnt": 1.0, "objs_jnt": 1.0, "goal": 1.0},
                'interact_site': "knob1_site"
            }
)