# Utility script to sync from robohive to myosuite repo repository
# Note: that its a one way sync at the moment "robohive => myosuite"
# Usage: .sync_myo.sh <full_path>/robohive/ <full_path>/myoSuite/

echo ==== Syncing $1 to $2 ====
src_path=$1  # Use absolute paths
dst_path=$2  # Use absolute paths
mjc_path=~/.mujoco

# convert xml to mjb
xml2mjb()
{
    # make directory if unavailable
    mkdir -p $dst_path/myosuite/envs/myo/assets/$1
    # remove old files if found
    rm -f $dst_path/myosuite/envs/myo/assets/$1/$2.mjb
    # compile into mjb
    ./mujoco210/bin/compile $src_path/robohive/envs/myo/assets/$1/$2.xml $dst_path/myosuite/envs/myo/assets/$1/$2.mjb
}

# Models
# cd $mjc_path
# xml2mjb arm myo_elbow_1dof6muscles_1dofexo
# xml2mjb arm myo_elbow_1dof6muscles
# xml2mjb basic myo_load
# xml2mjb finger motor_finger_v0
# xml2mjb finger myo_finger_v0
# xml2mjb hand myo_hand_baoding
# xml2mjb hand myo_hand_hold
# xml2mjb hand myo_hand_keyturn
# xml2mjb hand myo_hand_pen
# xml2mjb hand myo_hand_pose
# xml2mjb hand myo_hand_die
cd $src_path
echo $PWD

# Envs
mkdir -p $2/myosuite/envs/myo
rsync -av --progress $src_path/robohive/envs/env_base.py $dst_path/myosuite/envs/
rsync -av --progress $src_path/robohive/envs/env_variants.py $dst_path/myosuite/envs/
rsync -av --progress $src_path/robohive/envs/obs_vec_dict.py $dst_path/myosuite/envs/

# Envs/Myo
rsync -av --progress $src_path/robohive/envs/myo/*.md $dst_path/myosuite/envs/myo/
rsync -av --progress $src_path/robohive/envs/myo/assets/* $dst_path/myosuite/envs/myo/assets

# MyoBase Envs
rsync -av --progress $src_path/robohive/envs/myo/myobase/ $dst_path/myosuite/envs/myo/myobase/

# MyoChallenge Envs
rsync -av --progress $src_path/robohive/envs/myo/myochallenge/*.py $dst_path/myosuite/envs/myo/myochallenge/

# MyoDM Envs
rsync -av --progress $src_path/robohive/envs/myo/myodm/* $dst_path/myosuite/envs/myo/myodm

# Robot
mkdir -p $2/myosuite/robot
rsync -av --progress $src_path/robohive/robot/robot.py $dst_path/myosuite/robot/

# Utils
mkdir -p $2/myosuite/utils
rsync -av --progress $src_path/robohive/utils/*.py $dst_path/myosuite/utils/

# Physics
mkdir -p $2/myosuite/physics
rsync -av --progress $src_path/robohive/physics/*.py $dst_path/myosuite/physics/

# renderer
mkdir -p $2/myosuite/renderer
rsync -av --progress $src_path/robohive/renderer/*.py $dst_path/myosuite/renderer/

# logger
mkdir -p $2/myosuite/logger
rsync -av --progress $src_path/robohive/logger/*.py $dst_path/myosuite/logger/

# Test
mkdir -p $2/myosuite/tests
rsync -av --progress $src_path/robohive/tests/test_envs.py $dst_path/myosuite/tests/
rsync -av --progress $src_path/robohive/tests/test_myo.py $dst_path/myosuite/tests/

# Replace
# sed -i "s/robohive\./myosuite\./g" $dst_path/myosuite/envs/myo/__init__.py
find $dst_path/myosuite -type f -name "*.py" -exec sed -i "s/robohive\./myosuite\./g" {} \;
find $dst_path/myosuite/tests -type f -name "*.py" -exec sed -i "s/robohive/myosuite/g" {} \;
find $dst_path/myosuite/logger -type f -name "examine_reference.py" -exec sed -i "s/robohive/myosuite/g" {} \;
find $dst_path/myosuite -type f -name "*.py" -exec sed -i "s/RoboHive:>/MyoSuite:>/g" {} \;

# configs
rsync -av --progress $src_path/.gitignore $dst_path/

# Clean unnecessary
rm $dst_path/myosuite/envs/myo/myobase/baoding_v1.py
# rm $dst_path/myosuite/envs/myo/myobase/reorient_v0.py