""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com), Sudeep Dasari (sdasari@andrew.cmu.edu )
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gym.envs.registration import register
import numpy as np
import os
import collections
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering TCDM Envs")

# Task specification format
task_spec = collections.namedtuple('task_spec',
        ['name',        # task_name
         'robot',       # robot name
         'object',      # object name
         'motion',      # motion reference file path
         ])

# Adroit tasks
Adroit_task_spec = (
    task_spec(name='AdroitAirplaneFly-v0', robot='Adroit', object='airplane', motion='/assets/data/Adroit_airplane_fly.npz'),
    task_spec(name='AdroitAirplanePass-v0', robot='Adroit', object='airplane', motion='/assets/data/Adroit_airplane_pass.npz'),
    task_spec(name='AdroitAlarmclockLift-v0', robot='Adroit', object='alarmclock', motion='/assets/data/Adroit_alarmclock_lift.npz'),
    task_spec(name='AdroitAlarmclockSee-v0', robot='Adroit', object='alarmclock', motion='/assets/data/Adroit_alarmclock_see.npz'),
    task_spec(name='AdroitBananaPass-v0', robot='Adroit', object='banana', motion='/assets/data/Adroit_banana_pass.npz'),
    task_spec(name='AdroitBinocularsPass-v0', robot='Adroit', object='binoculars', motion='/assets/data/Adroit_binoculars_pass.npz'),
    task_spec(name='AdroitCupDrink-v0', robot='Adroit', object='cup', motion='/assets/data/Adroit_cup_drink.npz'),
    task_spec(name='AdroitCupPour-v0', robot='Adroit', object='cup', motion='/assets/data/Adroit_cup_pour.npz'),
    task_spec(name='AdroitDuckInspect-v0', robot='Adroit', object='duck', motion='/assets/data/Adroit_duck_inspect.npz'),
    task_spec(name='AdroitDuckLift-v0', robot='Adroit', object='duck', motion='/assets/data/Adroit_duck_lift.npz'),
    task_spec(name='AdroitElephantPass-v0', robot='Adroit', object='elephant', motion='/assets/data/Adroit_elephant_pass.npz'),
    task_spec(name='AdroitEyeglassesPass-v0', robot='Adroit', object='eyeglasses', motion='/assets/data/Adroit_eyeglasses_pass.npz'),
    task_spec(name='AdroitFlashlightLift-v0', robot='Adroit', object='flashlight', motion='/assets/data/Adroit_flashlight_lift.npz'),
    task_spec(name='AdroitFlashlightOn-v0', robot='Adroit', object='flashlight', motion='/assets/data/Adroit_flashlight_on.npz'),
    task_spec(name='AdroitFlutePass-v0', robot='Adroit', object='flute', motion='/assets/data/Adroit_flute_pass.npz'),
    task_spec(name='AdroitFryingpanCook-v0', robot='Adroit', object='fryingpan', motion='/assets/data/Adroit_fryingpan_cook.npz'),
    task_spec(name='AdroitHammerUse-v0', robot='Adroit', object='hammer', motion='/assets/data/Adroit_hammer_use.npz'),
    task_spec(name='AdroitHandInspect-v0', robot='Adroit', object='hand', motion='/assets/data/Adroit_hand_inspect.npz'),
    task_spec(name='AdroitHeadphonesPass-v0', robot='Adroit', object='headphones', motion='/assets/data/Adroit_headphones_pass.npz'),
    task_spec(name='AdroitKnifeChop-v0', robot='Adroit', object='knife', motion='/assets/data/Adroit_knife_chop.npz'),
    task_spec(name='AdroitLightbulbPass-v0', robot='Adroit', object='lightbulb', motion='/assets/data/Adroit_lightbulb_pass.npz'),
    task_spec(name='AdroitMouseLift-v0', robot='Adroit', object='mouse', motion='/assets/data/Adroit_mouse_lift.npz'),
    task_spec(name='AdroitMouseUse-v0', robot='Adroit', object='mouse', motion='/assets/data/Adroit_mouse_use.npz'),
    task_spec(name='AdroitMugDrink3-v0', robot='Adroit', object='mug', motion='/assets/data/Adroit_mug_drink3.npz'),
    task_spec(name='AdroitPiggybankUse-v0', robot='Adroit', object='piggybank', motion='/assets/data/Adroit_piggybank_use.npz'),
    task_spec(name='AdroitScissorsUse-v0', robot='Adroit', object='scissors', motion='/assets/data/Adroit_scissors_use.npz'),
    task_spec(name='AdroitSpheremediumLift-v0', robot='Adroit', object='spheremedium', motion='/assets/data/Adroit_spheremedium_lift.npz'),
    task_spec(name='AdroitStampStamp-v0', robot='Adroit', object='stamp', motion='/assets/data/Adroit_stamp_stamp.npz'),
    task_spec(name='AdroitStanfordbunnyInspect-v0', robot='Adroit', object='stanfordbunny', motion='/assets/data/Adroit_stanfordbunny_inspect.npz'),
    task_spec(name='AdroitStaplerLift-v0', robot='Adroit', object='stapler', motion='/assets/data/Adroit_stapler_lift.npz'),
    task_spec(name='AdroitToothbrushLift-v0', robot='Adroit', object='toothbrush', motion='/assets/data/Adroit_toothbrush_lift.npz'),
    task_spec(name='AdroitToothpasteLift-v0', robot='Adroit', object='toothpaste', motion='/assets/data/Adroit_toothpaste_lift.npz'),
    task_spec(name='AdroitToruslargeInspect-v0', robot='Adroit', object='toruslarge', motion='/assets/data/Adroit_toruslarge_inspect.npz'),
    task_spec(name='AdroitTrainPlay-v0', robot='Adroit', object='train', motion='/assets/data/Adroit_train_play.npz'),
    task_spec(name='AdroitTrainPlay1-v0', robot='Adroit', object='train', motion='/assets/data/Adroit_train_play1_old.npz'),
    task_spec(name='AdroitWatchLift-v0', robot='Adroit', object='watch', motion='/assets/data/Adroit_watch_lift.npz'),
    task_spec(name='AdroitWaterbottleLift-v0', robot='Adroit', object='waterbottle', motion='/assets/data/Adroit_waterbottle_lift.npz'),
    task_spec(name='AdroitWaterbottleShake-v0', robot='Adroit', object='waterbottle', motion='/assets/data/Adroit_waterbottle_shake.npz'),
    task_spec(name='AdroitWineglassDrink1-v0', robot='Adroit', object='wineglass', motion='/assets/data/Adroit_wineglass_drink1.npz'),
    task_spec(name='AdroitWineglassDrink2-v0', robot='Adroit', object='wineglass', motion='/assets/data/Adroit_wineglass_drink2.npz'),
)

# Register Adroit envs using reference motion
def register_adroit_object_trackref(task_name, object_name, motion_path=None):
    # Track reference motion
    # print("'"+task_name+"'", end=", ")
    register(
        id=task_name,
        entry_point='robohive.envs.tcdm.track:TrackEnv',
        max_episode_steps=75, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/assets/Adroit_object.xml',
                'object_name': object_name,
                'reference':curr_dir+motion_path,
            }
    )
for task_name, robot_name, object_name, motion_path in Adroit_task_spec:
    register_adroit_object_trackref(task_name, object_name, motion_path)



# All available objects
OBJECTS = ('airplane','alarmclock','apple','banana','binoculars','bowl','camera','coffeemug','cubelarge','cubemedium','cubemiddle','cubesmall','cup','cylinderlarge','cylindermedium','cylindersmall','doorknob','duck','elephant','eyeglasses','flashlight','flute','fryingpan','gamecontroller','hammer','hand','headphones','human','knife','lightbulb','mouse','mug','phone','piggybank', 'pyramidlarge','pyramidmedium','pyramidsmall','rubberduck','scissors','spherelarge','spheremedium','spheresmall','stamp','stanfordbunny','stapler','table','teapot','toothbrush','toothpaste','toruslarge','torusmedium','torussmall','train','watch','waterbottle','wineglass','wristwatch')

# Register object envs
def register_Adroit_object(object_name):
    task_name = 'Adroit{}Fixed-v0'.format(object_name.title())
    # print("'"+task_name+"'", end=", ")

    # Envs with fixed target
    register(
        id=task_name,
        entry_point='robohive.envs.tcdm.track:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/assets/Adroit_object.xml',
                'object_name': object_name,
                'reference': {'time':(0.0, 4.0),
                                'robot':np.zeros((1, 30)),
                                'robot_vel':np.zeros((1,30)),
                                'object_init':np.array((-.2, -.2, 0.0, 1.0, 0.0, 0.0, 0.0)),
                                'object':np.reshape(np.array((.2, .2, 0.0, 1.0, 0.0, 0.0, 0.0)), (1,7))
                            }
            }
    )

    # Envs with random target
    task_name = 'Adroit{}Random-v0'.format(object_name.title())
    # print("'"+task_name+"'", end=", ")
    register(
        id=task_name,
        entry_point='robohive.envs.tcdm.track:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/assets/Adroit_object.xml',
                'object_name': object_name,
                'reference': {'time':(0.0, 4.0),
                                'robot':np.zeros((2, 30)),
                                'robot_vel':np.zeros((2, 30)),
                                'object':np.array([ [-.2, -.2, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                    [0.2, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0]])
                            }
            }
    )
for obj in OBJECTS:
    register_Adroit_object(obj)



# Register Single object env
def register_Franka_object(object_name, data_path=None):
    # Track reference motion
    register(
        id='Franka{}TrackFixed-v0'.format(object_name.title()),
        entry_point='robohive.envs.tcdm.track:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/assets/Franka_object.xml',
                # 'data_path': curr_dir+data_path,
                'object_name': object_name,
                # 'reference': None, # TODO: pass reference trajectories
                'reference': {'time':0.0,
                                'robot':np.zeros((1,9)),
                                'robot_vel':np.zeros((1,30)),
                                'object':np.ones((1,7))}
            }
    )
for obj in OBJECTS:
    register_Franka_object(obj, data_path=None)

