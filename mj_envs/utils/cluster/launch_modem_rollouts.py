"""
This is a launcher script for launching mjrl training using hydra
"""

import os
import time as timer
import hydra
from omegaconf import DictConfig, OmegaConf
from modem_rollout_script import collect_rollouts

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="default", config_path="config")
def configure_jobs(job_data):
    OmegaConf.resolve(job_data) # resolve configs
    
    collect_rollouts(job_data)

if __name__ == "__main__":
    configure_jobs()
