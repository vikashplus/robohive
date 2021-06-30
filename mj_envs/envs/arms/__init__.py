from gym.envs.registration import register


# FRANKA =======================================================================

# Reach to fixed target
register(
    id='FrankaReachFixed-v0',
    entry_point='mj_envs.envs.arms.franka.reach_v0:FrankaReachFixed',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
)
from mj_envs.envs.arms.franka.reach_v0 import FrankaReachFixed

# Reach to random target
register(
    id='FrankaReachRandom-v0',
    entry_point='mj_envs.envs.arms.franka.reach_v0:FrankaReachRandom',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
)
from mj_envs.envs.arms.franka.reach_v0 import FrankaReachRandom

# Reach to fixed target
register(
    id='FrankaRelocateBoxFixed-v0',
    entry_point='mj_envs.envs.arms.franka.relocate_box_v0:FrankaRelocateBoxFixed',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
)
from mj_envs.envs.arms.franka.relocate_box_v0 import FrankaRelocateBoxFixed

# Reach to random target
register(
    id='FrankaRelocateBoxRandom-v0',
    entry_point='mj_envs.envs.arms.franka.relocate_box_v0:FrankaRelocateBoxRandom',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
)
from mj_envs.envs.arms.franka.relocate_box_v0 import FrankaRelocateBoxRandom

# FETCH =======================================================================

# Reach to fixed target
register(
    id='FetchReachFixed-v0',
    entry_point='mj_envs.envs.arms.fetch.reach_v0:FetchReachFixed',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
)
from mj_envs.envs.arms.fetch.reach_v0 import FetchReachFixed

# Reach to random target
register(
    id='FetchReachRandom-v0',
    entry_point='mj_envs.envs.arms.fetch.reach_v0:FetchReachRandom',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
)
from mj_envs.envs.arms.fetch.reach_v0 import FetchReachRandom
