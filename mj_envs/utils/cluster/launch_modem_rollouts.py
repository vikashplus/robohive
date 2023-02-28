"""
This is a launcher script for launching mjrl training using hydra
"""

import os
import time as timer
import hydra
from omegaconf import DictConfig, OmegaConf
from mj_envs.utils.collect_modem_rollouts import collect_rollouts

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
# Run locally
# python launch_modem_rollouts.py

# Generate 1000 demos on cluster
# python launch_modem_rollouts.py -m hydra/launcher=slurm hydra/output=slurm seed=0,100,200,300,400,500,600,700,800,900

@hydra.main(config_name="default", config_path="config")
def configure_jobs(job_data):
    OmegaConf.resolve(job_data) # resolve configs
    
    collect_rollouts(env_name=job_data.env_name, 
                     mode=job_data.mode, 
                     seed=job_data.seed, 
                     render=job_data.render, 
                     camera_name=job_data.camera_name, 
                     output_dir=job_data.output_dir, 
                     output_name=job_data.output_name, 
                     num_rollouts=job_data.num_rollouts,
                     sparse_reward=job_data.sparse_reward,
                     policy_path=job_data.policy_path)

if __name__ == "__main__":
    configure_jobs()
