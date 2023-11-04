from gym.envs.registration import register
from robohive.envs.env_variants import register_env_variant
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Quadruped Envs")

WALK_HORIZON = 160

# Reach to fixed target
register(
    id='DKittyWalkFixed-v0',
    entry_point='robohive.envs.quadrupeds.walk_v0:WalkBaseV0',
    max_episode_steps=WALK_HORIZON, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/dkitty/dkitty_walk_v0.xml',
        'config_path': curr_dir+'/dkitty/dkitty_walk_v0.config',
        'dof_range_names':('A:FRJ10', 'A:BRJ42'),
        'act_range_names':('A:FRJ10', 'A:BRJ42')
    }
)

# Reach to random target
register_env_variant(
    variant_id='DKittyWalkRandom-v0',
    env_id='DKittyWalkFixed-v0',
    variants={
            'target_distance_range':(1.75, 2.25),
            'target_angle_range':(np.pi/2-np.pi/6, np.pi/2+np.pi/6),
    },
    silent=True
)

# Reach to random target using proprio and visual inputs
from robohive.envs.quadrupeds.walk_v0 import WalkBaseV0
register_env_variant(
    variant_id='DKittyWalkRandom_v2d-v0',
    env_id='DKittyWalkRandom-v0',
    variants={
            "visual_keys": WalkBaseV0.DEFAULT_VISUAL_KEYS,
            # override the obs to avoid accidental leakage of oracle state info while using the visual envs
            # using time as dummy obs. time keys are added twice to avoid unintended singleton expansion errors.
            # Use proprioceptive data if needed - proprio_keys to configure, env.get_proprioception() to access
            "obs_keys": ['time', 'time']
    },
    silent=True
)



ORIENT_HORIZON = 80
# Orient to fixed orientation
register(
    id='DKittyOrientFixed-v0',
    entry_point='robohive.envs.quadrupeds.orient_v0:OrientBaseV0',
    max_episode_steps=ORIENT_HORIZON, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/dkitty/dkitty_walk_v0.xml',
        'config_path': curr_dir+'/dkitty/dkitty_walk_v0.config',
        'dof_range_names':('A:FRJ10', 'A:BRJ42'),
        'act_range_names':('A:FRJ10', 'A:BRJ42'),
        'target_distance_range':(0, 0),
        'target_angle_range':(-np.pi/2, -np.pi/2),
    }
)

# Orient to random orientation
register_env_variant(
    variant_id='DKittyOrientRandom-v0',
    env_id='DKittyOrientFixed-v0',
    variants={
            'target_distance_range':(0, 0),
            'target_angle_range':(-np.pi/6, 4*np.pi/3), # Robel was (-2*np.pi/3, -np.pi/6)
    },
    silent=True
)
# Orient to random orientation using proprio and visual inputs
from robohive.envs.quadrupeds.orient_v0 import OrientBaseV0
register_env_variant(
    variant_id='DKittyOrientRandom_v2d-v0',
    env_id='DKittyOrientRandom-v0',
    variants={
            "visual_keys": OrientBaseV0.DEFAULT_VISUAL_KEYS,
            # override the obs to avoid accidental leakage of oracle state info while using the visual envs
            # using time as dummy obs. time keys are added twice to avoid unintended singleton expansion errors.
            # Use proprioceptive data if needed - proprio_keys to configure, env.get_proprioception() to access
            "obs_keys": ['time', 'time']
    },
    silent=True
)



STAND_HORIZON = 80
# Stand-Up
register(
    id='DKittyStandFixed-v0',
    entry_point='robohive.envs.quadrupeds.stand_v0:StandBaseV0',
    max_episode_steps=STAND_HORIZON, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/dkitty/dkitty_walk_v0.xml',
        'config_path': curr_dir+'/dkitty/dkitty_stand_v0.config',
        'dof_range_names':('A:FRJ10', 'A:BRJ42'),
        'act_range_names':('A:FRJ10', 'A:BRJ42')
    }
)
# Stand-Up from random pose
register_env_variant(
    variant_id='DKittyStandRandom-v0',
    env_id='DKittyStandFixed-v0',
    variants={
        'reset_type': 'random'
    },
    silent=True
)
# Stand-Up from random pose using proprio and visual inputs
from robohive.envs.quadrupeds.orient_v0 import OrientBaseV0
register_env_variant(
    variant_id='DKittyStandRandom_v2d-v0',
    env_id='DKittyStandRandom-v0',
    variants={
            "visual_keys": OrientBaseV0.DEFAULT_VISUAL_KEYS,
            "obs_keys": ['time', 'time']
    },
    silent=True
)
