`RoboHive` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

0. We recommend installaition within a conda environement. If you don't have one yet, create one using
   ```
   conda create -n robohive python=3.8
   conda activate robohive
   ```

1. Robohive can be installed directly from the [PyPi](https://pypi.org/project/robohive/). Additional flags can be supplied to use additional  features
   ``` bash
   # only mujoco as available sim_backend
   pip install robohive
   # mujoco_py+mujoco as available sim_backend
   pip install robohive[mujoco_py]
   # mujoco_py+mujoco+visual encoders
   pip install robohive[mujoco_py, encoder]
   ```
   RoboHive will throw informative errors if any of these packages are invoked but not installed
   ```bash
   # Initialize robohive (one time)
   robohive_init
   ```

2. For **editable installation**, clone this repo (on a [tag](https://github.com/vikashplus/robohive/releases)) with pre-populated submodule dependencies as -
   ```
   git clone --branch <tag_name/branch_name> --recursive https://github.com/vikashplus/robohive.git
   ```
   Note: RoboHive agressively uses [git-submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Therefore, it is important to use the exact command above for installation. Basic understanding of how to work with submodules is expected.

   ```bash
   # only mujoco as available sim_backend
   $ pip install -e robohive
   # mujoco_py+mujoco as available sim_backend
   $ pip install -e robohive[mujoco_py]
   # mujoco_py+mujoco+visual encoder
   pip install -e robohive[mujoco_py, encoder]
   # with a0 binding for realworld robot
   $ pip install -e robohive[a0]
   ```
   <!-- **OR** Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
   ```bash
   export PYTHONPATH="<path/to/robohive>:$PYTHONPATH"
   ```-->
   RoboHive will throw informative errors if any of these packages are invoked but not installed
   ```bash
   # Initialize robohive (one time)
   robohive_init
   ```

3. To use `R3M`, `RRL`, `VC` as visual encoders.
   - For PyPI installation: Encoders can't be directly installated from PyPi as these encoders don't have PyPi packages. (we are working with the authors for a fix). To use the encoders, simply try using the encoders and follow the instructions on the console
   - For editable installation
      Open [setup.py](/setup.py), uncomment the `r3m` and `vc_models` dependency under `encoder` and then run
   ```
      # all visual encoders
      pip install robohive[encoder]
   ```

4. You can visualize the environments with random controls using the below command
   ```
   $ python robohive/utils/examine_env.py -e FrankaReachRandom-v0
   ```


# Installation FAQs
The package relies on `mujoco-py` which might be the trickiest part of the installation. See `known issues` below and also instructions from the mujoco-py [page](https://github.com/openai/mujoco-py) if you are stuck with mujoco-py installation.

## Linux
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```

- Visualization in linux: If the linux system has a GPU, then mujoco-py does not automatically preload the correct drivers. We added an alias `MJPL` in bashrc (see instructions) which stands for mujoco pre-load. When runing any python script that requires rendering, prepend the execution with MJPL.

   - Update `bashrc` by adding the following lines and source it
      ```
      alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
      ```
   - Use this preload when launching your scripts
      ```
      $ MJPL python script.py
      ```
- If there are issues with install of pytorch, please follow instructions from the [pytorch website](https://pytorch.org/) to install it properly based on the specific version of CUDA (or CPU-only) you have.

- If you encounter a patchelf error in mujoco_py install, you can fix this with the following command when inside the anaconda env: `conda install -c anaconda patchelf`. See this [page](https://github.com/openai/mujoco-py/issues/147) for additional info.

## Mac OS

- If there are issues with install of pytorch, please follow instructions from the [pytorch website](https://pytorch.org/) to install it properly.

- If you encounter a patchelf error in mujoco_py install, you can fix this with the following command when inside the anaconda env: `conda install -c anaconda patchelf`. See this [page](https://github.com/openai/mujoco-py/issues/147) for additional info.

- GCC error in Mac OS: If you get a GCC error from mujoco-py, you can get the correct version mujoco-py expects with `brew install gcc --without-multilib`. This may require uninstalling other versions of GCC that may have been previously installed with `brew remove gcc@6` for example. You can see which brew packages were already installed with `brew list`.


## Known Issues
- Errors related to osmesa during installation. This is a `mujoco-py` build error and would likely go away if the following command is used before creating the conda environment. If the problem still persists, please contact the developers of mujoco-py
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```
