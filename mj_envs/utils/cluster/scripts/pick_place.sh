#!/bin/bash

#python launch_modem_rollouts.py \
#    env_name=FrankaPickPlaceRandom-v0 \
#    mode=evaluation \
#    seed=1 \
#    num_rollouts=5 \
#    output_dir=/tmp/modem_collect \
#    render=offscreen \
#    camera_name=left_cam

#python launch_modem_rollouts.py -m \
#    env_name=FrankaPickPlaceRandom-v0 \
#    mode=evaluation \
#    seed=400,500,600,700,800,900 \
#    num_rollouts=100 \
#    output_dir=/checkpoint/plancaster/outputs/modem/demonstrations/franka-FrankaPickPlaceRandom \
#    hydra/launcher=slurm \
#    hydra/output=slurm 

python launch_modem_rollouts.py -m \
    env_name=FrankaPickPlaceRandom_v2d-v0 \
    mode=evaluation \
    seed=0,100,200,300,400,500,600,700,800,900 \
    num_rollouts=100 \
    output_dir=/checkpoint/plancaster/outputs/modem/demonstrations/franka-FrankaPickPlaceRandom_v2d \
    hydra/launcher=slurm \
    hydra/output=slurm 