# Human demonstrations
Human demos for the kitchen scene are collected using [MuJoCo-VR](https://github.com/vikashplus/puppet) and are stored in [MuJoCo's `.mjl` log format](http://www.mujoco.org/book/haptix.html#uiRecord). The demonstration dataset is available as a zip file here [kitchen_demos_multitask.zip](https://github.com/google-research/relay-policy-learning/raw/master/kitchen_demos_multitask.zip). Download and unzip the demonstration in the current `assets` folder. Below we provide instructions to parse these demonstrations in accordance with the kitchen environments.

# Instructions on how to parse the demos
1. Download and unzip the demonstration file
2. Clone and add [MuJoCo-VR](https://github.com/vikashplus/puppet) to your PYTHONPATH
3. You can render/playback the demo dataset on an env using
```python robohive/envs/relay_kitchen/assets/parse_demos.py -e kitchen-v2 --demo_dir robohive/envs/relay_kitchen/assets/kitchen_demos_multitask/SELECTED_DEMO_DIR/ --view playback --skip 40```