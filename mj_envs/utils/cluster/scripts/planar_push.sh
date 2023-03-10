#!/bin/bash

#python launch_modem_rollouts.py \
#    env_name=FrankaPlanarPush-v0 \
#    mode=evaluation \
#    seed=0 \
#    num_rollouts=1000 \
#    output_dir=/checkpoint/plancaster/outputs/modem/demonstrations/franka-FrankaPlanarPush \
#    policy_path=/private/home/plancaster/modem/hand_dapg/dapg/examples/planar_push_exp/iterations/best_policy.pickle \


python launch_modem_rollouts.py -m \
    env_name=FrankaPlanarPush_v2d-v0 \
    mode=evaluation \
    seed=0,100,200,300,400,500,600,700,800,900 \
    num_rollouts=100 \
    output_dir=/checkpoint/plancaster/outputs/modem/demonstrations/franka-FrankaPlanarPush_v2d \
    policy_path=/private/home/plancaster/modem/hand_dapg/dapg/examples/planar_push_exp/iterations/best_policy.pickle \
    hydra/launcher=slurm \
    hydra/output=slurm 