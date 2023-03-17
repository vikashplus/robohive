"""
Script to process reference data into reference format
reference = collections.namedtuple('reference',
        ['time',        # int
         'robot',       # shape(N, n_robot_jnt) ==> robot trajectory
         'object',      # shape(M, n_objects_jnt) ==> object trajectory
         'robot_init',  # shape(n_objects_jnt) ==> initial robot pose
         'object_init'  # shape(n_objects_jnt) ==> initial object
         ])
"""
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))


sim_dt = 0.002
n_robot_jnt = 30

def load(reference_path):
    return {k:v for k, v in np.load(reference_path).items()}

def proces_n_save(data, new_path_name):
    print(f"Motion length = {data['length']}")
    time = np.arange(0, data['length']) * data['SIM_SUBSTEPS']*sim_dt
    data_obj = np.concatenate([data['object_translation'], data['object_orientation']], axis=1)
    np.savez(new_path_name,
             time=time,
             robot=data['s_0'][None,:n_robot_jnt],
             object=data_obj,
             robot_int=data['s_0'][:n_robot_jnt],
             object_int=data['s_0'][n_robot_jnt:],
             )

file_names = [
    # old_path, new_path
    # ('airplane_fly1.npz',   'Adroit_airplane_fly.npz'),
    # ('airplane_pass1.npz',  'Adroit_airplane_pass.npz'),
    # ('alarmclock_lift.npz', 'Adroit_alarmclock_lift.npz'),
    # ('alarmclock_see1.npz', 'Adroit_alarmclock_see.npz'),
    # ('banana_pass1.npz',    'Adroit_banana_pass.npz'),

    ('airplane_fly1.npz', 'Adroit_airplane_fly.npz'),
    ('airplane_pass1.npz', 'Adroit_airplane_pass.npz'),
    ('alarmclock_lift.npz', 'Adroit_alarmclock_lift.npz'),
    ('alarmclock_see1.npz', 'Adroit_alarmclock_see.npz'),
    ('banana_pass1.npz', 'Adroit_banana_pass.npz'),
    ('binoculars_pass1.npz', 'Adroit_binoculars_pass.npz'),
    ('cup_drink1.npz', 'Adroit_cup_drink.npz'),
    ('cup_pour1.npz', 'Adroit_cup_pour.npz'),
    ('duck_inspect1.npz', 'Adroit_duck_inspect.npz'),
    ('duck_lift.npz', 'Adroit_duck_lift.npz'),
    ('elephant_pass1.npz', 'Adroit_elephant_pass.npz'),
    ('eyeglasses_pass1.npz', 'Adroit_eyeglasses_pass.npz'),
    ('flashlight_lift.npz', 'Adroit_flashlight_lift.npz'),
    ('flashlight_on2.npz', 'Adroit_flashlight_on.npz'),
    ('flute_pass1.npz', 'Adroit_flute_pass.npz'),
    ('fryingpan_cook2.npz', 'Adroit_fryingpan_cook.npz'),
    ('hammer_use1.npz', 'Adroit_hammer_use.npz'),
    ('hand_inspect1.npz', 'Adroit_hand_inspect.npz'),
    ('headphones_pass1.npz', 'Adroit_headphones_pass.npz'),
    ('knife_chop1.npz', 'Adroit_knife_chop.npz'),
    ('lightbulb_pass1.npz', 'Adroit_lightbulb_pass.npz'),
    ('mouse_lift.npz', 'Adroit_mouse_lift.npz'),
    ('mouse_use1.npz', 'Adroit_mouse_use.npz'),
    ('mug_drink3.npz', 'Adroit_mug_drink3.npz'),
    ('piggybank_use1.npz', 'Adroit_piggybank_use.npz'),
    ('scissors_use1.npz', 'Adroit_scissors_use.npz'),
    ('spheremedium_lift.npz', 'Adroit_spheremedium_lift.npz'),
    ('stamp_stamp1.npz', 'Adroit_stamp_stamp.npz'),
    ('stanfordbunny_inspect1.npz', 'Adroit_stanfordbunny_inspect.npz'),
    ('stapler_lift.npz', 'Adroit_stapler_lift.npz'),
    ('toothbrush_lift.npz', 'Adroit_toothbrush_lift.npz'),
    ('toothpaste_lift.npz', 'Adroit_toothpaste_lift.npz'),
    ('toruslarge_inspect1.npz', 'Adroit_toruslarge_inspect.npz'),
    ('train_play1.npz', 'Adroit_train_play.npz'),
    ('train_play1_old.npz', 'Adroit_train_play1_old.npz'),
    ('watch_lift.npz', 'Adroit_watch_lift.npz'),
    ('waterbottle_lift.npz', 'Adroit_waterbottle_lift.npz'),
    ('waterbottle_shake1.npz', 'Adroit_waterbottle_shake.npz'),
    ('wineglass_drink1.npz', 'Adroit_wineglass_drink1.npz'),
    ('wineglass_drink2.npz', 'Adroit_wineglass_drink2.npz'),
]

# path_name = curr_dir+'/banana_pass1.npz'
# new_path_name = curr_dir+'/Adroit_banana_pass1.npz'
old_path_dir = '/Users/vikashplus/Libraries/mimic/trajectories'
for old_name, new_name in file_names:
    print(f"{new_name}")
    old_path = os.path.join(old_path_dir, old_name)
    new_path = os.path.join(curr_dir, new_name)
    print(f"Processing: {old_path} as {new_path}", end=" :: ")
    data = load(old_path)
    proces_n_save(data, new_path)


