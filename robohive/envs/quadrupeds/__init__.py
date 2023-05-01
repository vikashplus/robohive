from gym.envs.registration import register
from robohive.envs.env_variants import register_env_variant
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Quadruped Envs")

WALK_HORIZON = 100

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
            'target_angle_range':(-np.pi/6, np.pi/6),
    }
)

# register(
#     id='DKittyWalkFixed-v0',
#     entry_point='darwin.darwin_envs.walk.dkitty.dkitty_walk_v0:DKittyWalkFixed',
#     max_episode_steps=WALK_HORIZON,  #2.5ms*10fs*100 = 2.5 sec
# )
# register(
#     id='DKittyWalkRandom-v0',
#     entry_point='darwin.darwin_envs.walk.dkitty.dkitty_walk_v0:DKittyWalkRandom',
#     max_episode_steps=WALK_HORIZON,  #2.5ms*10fs*100 = 2.5 sec
# )
# register(
#     id='DKittyWalkRandomDynamics-v0',
#     entry_point='darwin.darwin_envs.walk.dkitty.dkitty_walk_v0:DKittyWalkRandomDynamics',
#     max_episode_steps=WALK_HORIZON,  #2.5ms*10fs*100 = 2.5 sec
# )