""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com), Sudeep Dasari (sdasari@andrew.cmu.edu )
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gym.envs.registration import register
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Register Single object env
def register_Adroit_object(object_name):
    # Track reference motion
    register(
        id='Adroit{}TrackFixed-v0'.format(object_name.title()),
        entry_point='mj_envs.envs.dexman.track:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/assets/Adroit_object.xml',
                'object_name': object_name,
                'target_pose': None, # TODO: pass reference trajectories
            }
    )

# Register Single object env
def register_Franka_object(object_name):
    # Track reference motion
    register(
        id='Franka{}TrackFixed-v0'.format(object_name.title()),
        entry_point='mj_envs.envs.dexman.track:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/assets/Franka_object.xml',
                'object_name': object_name,
                'target_pose': None, # TODO: pass reference trajectories
            }
    )

# Register all object envs
OBJECTS = ('airplane','alarmclock','apple','banana','binoculars','bowl','camera','coffeemug','cubelarge','cubemedium','cubemiddle','cubesmall','cup','cylinderlarge','cylindermedium','cylindersmall','doorknob','duck','elephant','eyeglasses','flashlight','flute','fryingpan','gamecontroller','hammer','hand','headphones','human','knife','lightbulb','mouse','mug','phone','piggybank', 'pyramidlarge','pyramidmedium','pyramidsmall','rubberduck','scissors','spherelarge','spheremedium','spheresmall','stamp','stanfordbunny','stapler','table','teapot','toothbrush','toothpaste','toruslarge','torusmedium','torussmall','train','watch','waterbottle','wineglass','wristwatch')

for obj in OBJECTS:
    register_Adroit_object(obj)
    register_Franka_object(obj)

