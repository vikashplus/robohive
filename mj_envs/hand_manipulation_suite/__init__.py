from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

# Pick up the knife
register(
    id='hand_knife-v0',
    entry_point='mj_envs.hand_manipulation_suite:KnifeEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.knife_v0 import KnifeEnvV0

# Pick up the exp_knife
register(
    id='hand_exp-v0',
    entry_point='mj_envs.hand_manipulation_suite:ExpEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.exp_v0 import ExpEnvV0

# Pick up the spatula
register(
    id='hand_spatula-v0',
    entry_point='mj_envs.hand_manipulation_suite:SpatulaEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.spatula_v0 import SpatulaEnvV0

# Pick up the screwDriver
register(
    id='hand_screwDriver-v0',
    entry_point='mj_envs.hand_manipulation_suite:ScrewDriverEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.screwDriver_v0 import ScrewDriverEnvV0

# Pick up the ratchet
register(
    id='hand_ratchet-v0',
    entry_point='mj_envs.hand_manipulation_suite:RatchetEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.ratchet_v0 import RatchetEnvV0

# Pick up the turner
register(
    id='hand_turner-v0',
    entry_point='mj_envs.hand_manipulation_suite:TurnerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.turner_v0 import TurnerEnvV0

# Pick up the hammer
register(
    id='hand_hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hand_hammer_v0 import HammerEnvV0

# Pick up the mug
register(
    id='hand_mug-v0',
    entry_point='mj_envs.hand_manipulation_suite:MugEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.mug_v0 import MugEnvV0

# Pick up the bottle
register(
    id='hand_bottle-v0',
    entry_point='mj_envs.hand_manipulation_suite:BottleEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.bottle_v0 import BottleEnvV0

# Pick up the pan
register(
    id='hand_pan-v0',
    entry_point='mj_envs.hand_manipulation_suite:PanEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.pan_v0 import PanEnvV0

# Pick up the lid
register(
    id='hand_lid-v0',
    entry_point='mj_envs.hand_manipulation_suite:LidEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.lid_v0 import LidEnvV0

