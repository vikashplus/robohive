# Human demonstrations
Human demos for the kitchen scene collected using [MuJoCo-VR](https://github.com/vikashplus/puppet) and are stored in [MuJoCo's `.mjl` log format](http://www.mujoco.org/book/haptix.html#uiRecord). The demonstration dataset is included with this repository as the a zip file named [kitchen_demos_multitask.zip](kitchen_demos_multitask.zip). Below we provide instructions to parse these demonstrations in accordance with the kitchen environments.

# Instructions on how to parse the demos
1. Download and unzip the demonstration file
2. Clone and add [MuJoCo-VR](https://github.com/vikashplus/puppet) to your PYTHONPATH
3. You can render/playback the demo dataset on an env using
```python mj_envs/envs/relay_kitchen/assets/parse_demos.py -e kitchen-v2 --demo_dir mj_envs/envs/relay_kitchen/assets/kitchen_demos_multitask/SELECTED_DEMO_DIR/ --view playback --skip 40```