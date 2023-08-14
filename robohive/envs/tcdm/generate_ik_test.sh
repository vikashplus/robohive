#!/bin/sh

mocap_path="/mimic_dataset_mocap/data/human_trajs/"
#python playback_mocap.py --human_motion_file /mimic_dataset_mocap/data/human_trajs/s1/airplane_pass_1.npz ./data_myoHand/ik/s1/airplane_lift_myo.npz --start 163 --end 484 --sim_name MyoHandAirplaneFixed-v0 --length 68 --no_viewer
echo "$mocap_path"
mkdir -p data_myoHand/
mkdir -p data_myoHand/ik
for i in {1..10}
do
    mkdir -p "data_myoHand/ik/s$i"
done
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/airplane_lift.npz ./data_myoHand/ik/s1/airplane_lift_myo.npz --start 163 --end 484 --sim_name MyoHandAirplaneFixed-v0 --length 68 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/airplane_pass_1.npz ./data_myoHand/ik/s1/airplane_pass_1_myo.npz --start 146 --end 697 --sim_name MyoHandAirplaneFixed-v0 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/alarmclock_lift.npz ./data_myoHand/ik/s1/alarmclock_lift_myo.npz --start 112 --end 380 --sim_name MyoHandAlarmclockFixed-v0 --length 57 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/alarmclock_pass_1.npz ./data_myoHand/ik/s1/alarmclock_pass_1_myo.npz --start 105 --end 675 --sim_name MyoHandAlarmclockFixed-v0 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/alarmclock_see_1.npz ./data_myoHand/ik/s1/alarmclock_see_1_myo.npz --start 156 --end 740 --sim_name MyoHandAlarmclockFixed-v0 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/apple_lift.npz ./data_myoHand/ik/s1/apple_lift_myo.npz --start 175 --end 448 --sim_name MyoHandAppleFixed-v0 --length 58 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/apple_pass_1.npz ./data_myoHand/ik/s1/apple_pass_1_myo.npz --start 115 --end 593 --sim_name MyoHandAppleFixed-v0 --no_viewer
nice -n 19 python playback_mocap.py --human_motion_file "$mocap_path"/s1/camera_pass_1.npz ./data_myoHand/ik/s1/camera_pass_1_myo.npz --start 164 --end 638 --sim_name MyoHandCameraFixed-v0 --no_viewer
