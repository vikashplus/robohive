""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gym.envs.registration import register
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
from robohive.envs.env_variants import register_env_variant

print("RoboHive:> Registering Claw Envs")

# TRIFINGER REORIENT =======================================================================
from robohive.envs.claws.reorient_v0 import ReorientBaseV0

# Reorient to fixed target
register(
    id='TrifingerReachFixed-v0',
    entry_point='robohive.envs.claws.reorient_v0:ReorientBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/trifinger/trifinger_reorient.xml',
        'object_site_name': "object",
        'target_site_name': "target",
        'target_xyz_range': {'high':[.01, .03, 0.95], 'low':[.01, .03, 0.95]},
        'target_euler_range': {'high':[.2, .3, .1], 'low':[.2, .3, .1]}
    }
)

# Reorient to random target
register(
    id='TrifingerReachRandom-v0',
    entry_point='robohive.envs.claws.reorient_v0:ReorientBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/trifinger/trifinger_reorient.xml',
        'object_site_name': "object",
        'target_site_name': "target",
        'target_xyz_range': {'high':[.05, .05, 0.9], 'low':[-.05, -.05, 0.99]},
        'target_euler_range': {'high':[1, 1, 1], 'low':[-1, -1, -1]}
    }
)

# Reach to random target using visual inputs
def register_visual_envs(env_id, encoder_type):
    register_env_variant(
        env_id=env_id,
        variant_id=env_id[:-3]+'_v'+encoder_type+env_id[-3:],
        variants={'obs_keys':
                    ['qp', 'qv'],
                'visual_keys':[
                    "rgb:center_cam:224x224:{}".format(encoder_type),]
        },
        silent=True
    )
for env in ['TrifingerReachFixed-v0', 'TrifingerReachRandom-v0']:
    for enc in ["r3m18", "r3m34", "r3m50", "rrl18", "rrl34", "rrl50", "2d"]:
        register_visual_envs(env, enc)
