# Utility script to sync from mj_envs to myosuite repo repository
# Note: that its a one way sync at the moment "mj_envs => myosuite"
# Usage: .sync_myo.sh <full_path>/mj_envs/ <full_path>/myoSuite/

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
    ./mujoco210/bin/compile $src_path/mj_envs/envs/myo/assets/$1/$2.xml $dst_path/myosuite/envs/myo/assets/$1/$2.mjb
}

# Models
cd $mjc_path
xml2mjb arm myo_elbow_1dof6muscles_1dofexo
xml2mjb arm myo_elbow_1dof6muscles
xml2mjb basic myo_load
xml2mjb finger motor_finger_v0
xml2mjb finger myo_finger_v0
xml2mjb hand myo_hand_baoding
xml2mjb hand myo_hand_hold
xml2mjb hand myo_hand_keyturn
xml2mjb hand myo_hand_pen
xml2mjb hand myo_hand_pose
xml2mjb hand myo_hand_die
cd $src_path
echo $PWD

# Envs
mkdir -p $2/myosuite/envs/myo
rsync -av --progress $src_path/mj_envs/envs/env_base.py $dst_path/myosuite/envs/
rsync -av --progress $src_path/mj_envs/envs/env_variants.py $dst_path/myosuite/envs/
rsync -av --progress $src_path/mj_envs/envs/myo/*.md $dst_path/myosuite/envs/myo/
rsync -av --progress $src_path/mj_envs/envs/myo/*.py $dst_path/myosuite/envs/myo/
rsync -av --progress $src_path/mj_envs/envs/myo/myochallenge/*.py $dst_path/myosuite/envs/myo/myochallenge/
sed -i '' "s/xml/mjb/g" $dst_path/myosuite/envs/myo/__init__.py
sed -i '' "s/xml/mjb/g" $dst_path/myosuite/envs/myo/myochallenge/__init__.py

# Robot
mkdir -p $2/myosuite/robot
rsync -av --progress $src_path/mj_envs/robot/robot.py $dst_path/myosuite/robot/

# Utils
mkdir -p $2/myosuite/utils
rsync -av --progress $src_path/mj_envs/utils/*.py $dst_path/myosuite/utils/

# Test
mkdir -p $2/myosuite/tests
rsync -av --progress $src_path/mj_envs/tests/test_envs.py $dst_path/myosuite/tests/
rsync -av --progress $src_path/mj_envs/tests/test_myo.py $dst_path/myosuite/tests/

# Replace
# sed -i '' "s/mj_envs/myosuite/g" $dst_path/myosuite/envs/myo/__init__.py
find $dst_path/myosuite -type f -name "*.py" -exec sed -i '' "s/mj_envs\./myosuite\./g" {} \;

# configs
rsync -av --progress $src_path/.gitignore $dst_path/
