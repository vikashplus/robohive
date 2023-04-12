`RoboHive` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

0. We recommend installaition within a conda environement. If you don't have one yet, create one using
   ```
   conda create -n robohive python=3.8
   conda activate robohive
   ```

1. Robohive can be installed directly from the [PyPi](https://pypi.org/project/robohive/). Additional flags can be supplied to use additional  features
   ``` bash
   # only mujoco_py as available sim_backend
   pip install robohive
   # mujoco_py+mujoco as available sim_backend
   pip install robohive[mujoco]
   # mujoco_py+mujoco+visual encoders
   pip install robohive[mujoco, encoders]
   ```
   RoboHive will throw informative errors if any of these packages are invokes but not installed

2. For editable installation, clone this repo (on a [tag](https://github.com/vikashplus/robohive/releases)) with pre-populated submodule dependencies as -
   ```
   git clone --branch <tag_name/branch_name> --recursive https://github.com/vikashplus/robohive.git
   ```
   Note: RoboHive agressively uses [git-submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Therefore, it is important to use the exact command above for installation. Basic understanding of how to work with submodules is expected.

   ```bash
   $ cd robohive
   # only mujoco_py as available sim_backend
   $ pip install -e .
   # mujoco_py+mujoco as available sim_backend
   $ pip install -e ".[mujoco]"
   # mujoco_py+mujoco+visual encoders
   pip install robohive[mujoco, encoders]
   # with a0 binding for realworld robot
   $ pip install -e ".[a0]"
   ```
   **OR** Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
   ```
   export PYTHONPATH="<path/to/robohive>:$PYTHONPATH"
   ```
   To use `R3M`, `RRL`, `VC` as visual encoders. Open [setup.py](/setup.py), uncomment the `r3m` and `vc_models` dependency under `encoder` and then run
   ```
   # all visual encoders
   pip install robohive[encoders]
   ```

3. You can visualize the environments with random controls using the below command
   ```
   $ python robohive/utils/examine_env.py -e FrankaReachRandom-v0
   ```