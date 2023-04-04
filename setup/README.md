`RoboHive` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

0. We recommend installaition within a conda environement. If you don't have one yet, create one using
   ```
   conda create -n robohive python=3
   conda activate robohive
   ```

1. Robohive can be installed directly from the [PiPy](https://pypi.org/project/robohive/) using
   ```
   pip install robohive
   ```

2. For editable installation, clone this repo on branch `branch_name` with pre-populated submodule dependencies as -
   ```
   git clone --branch <branch_name> --recursive https://github.com/vikashplus/robohive.git
   ```
   Note: RoboHive agressively uses [git-submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Therefore, it is important to use the exact command above for installation. Basic understanding of how to work with submodules is expected.

   ```
   $ cd robohive
   $ pip install -e .[a0] #with a0 binding for realworld robot
   $ pip install -e .     #simulation only
   ```
   **OR** Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
   ```
   export PYTHONPATH="<path/to/robohive>:$PYTHONPATH"
   ```
3. You can visualize the environments with random controls using the below command
   ```
   $ python robohive/utils/examine_env.py -e FrankaReachRandom-v0
   ```