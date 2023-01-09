# Robot interface
RoboHive uses an abstract class called `robot` to interface with all agents/devices. Goal of the `robot` class is to
1. Abstract out the difference between a `simulation` vs `hardware` instantiation of our agents.
2. Provide a unified interface to specify operational+safety [configurations]().

![Alt text](robohive_robot_overview.png "Optional title")

RoboHive's goal is to make hardware as seamless to use as simulations. In order to achieve this, RoboHive projects hardware into simulation. Conceptually, this implies that all our computations happen as if we are always working in a simulated setting, hardware merely is a dynamic function that updates the sim. This has a few fundamental benefits
1. Interpretability of the simulation carried over to hardware. Users don't need to know all the hardware details.
2. It's easy to prototype in simulation and later port results into hardware. This is very useful for paradigms like sim2real.
3. Simulation provides a general parameterization to the hardware details when it's important.
4. Simulation rendering can be used as a real time visualizer for the hardware updates.

## Simulated Robot
Robohive uses MuJoCo as its simulated backbone. Since RoboHive operates in the simulation space, one can directly interact with our simulation for any computation/update.

## Hardware Robot
RoboHive provides a unified [base class](hardware_base.py) for all hardware interfaces. RoboHive comes pre-packaged with a few hardware interfaces for a few common robots. It's also easy to add a new interface. Interfaces are picked and configured using a [configuration file](). Goal of the hardware interface is to be the background process that pulls and pushes all updates to the simulation and hides all hardware details from the RoboHive users.

1. [Franke Emika Arm](https://www.franka.de/)
RoboHive uses [Polymetis](https://facebookresearch.github.io/fairo/polymetis/) for interfacing with Franka Arms. Please follow the install instructions provided by the Polymetis authors.

2. [Dynamixel](http://www.dynamixel.com/)
RoboHive supports all dynamixel based robots such as [ROBEL](http://roboticsbenchmarks.org/). Please follow driver install instructions [here](https://github.com/vikashplus/dynamixel).

3. [OptiTrack](https://optitrack.com/)
RoboHive supports optitrack motion tracking system. Please follow install instructions [here](https://github.com/vikashplus/OptiTrack)


4. [Robotiq Grippers](https://robotiq.com)
RoboHive uses [Polymetis](https://facebookresearch.github.io/fairo/polymetis/) for interfacing with robotiq grippers. Please follow the install instructions provided by the Polymetis authors.

5. [Real-Sense cameras](https://www.intelrealsense.com/)
Robohive relies on a pub-sub framework from [Fairo](https://github.com/facebookresearch/fairo) to interact with intel realsense cameras. Quick install
to get the a0 dependencies `pip install alephzero`

## Configuration
Robot configurations are specified using config files. The file is essentially a nested dict that specified details about the robot's `interface`, `sensor`, `actuator`, and `cams` configurations. See below for an example.
```
{
# device1:  interface, sensors, actuators, cams configs
'franka':{
   'interface': {'type': 'franka', 'ip_address':'172.16.0.1', 'gain_scale':0.5},
   'sensor':[
       {'range':(-2.9, 2.9), 'noise':0.05, 'hdr_id':0, 'scale':1, 'offset':0, 'name':'fr_arm_jp1'},
       {'range':(-1.8, 1.8), 'noise':0.05, 'hdr_id':1, 'scale':1, 'offset':0, 'name':'fr_arm_jp2'},
   ],
   'actuator':[
       {'pos_range':(-2.9, 2.9), 'vel_range':(-2, 2), 'hdr_id':0, 'scale':1, 'offset':0, 'name':'panda0_joint1'},
       {'pos_range':(-1.8, 1.8), 'vel_range':(-2, 2), 'hdr_id':1, 'scale':1, 'offset':0, 'name':'panda0_joint2'},
   ],
   'cam': []
},

# device2: interface, sensors, actuators, cams configs
'robotiq':{
   'interface': {'type': 'robotiq', 'ip_address':'172.16.0.1'},
   'sensor':[
       {'range':(0, 0.834), 'noise':0.0, 'hdr_id':0, 'name':'robotiq_2f_85', 'scale':-9.81, 'offset':0.834},
   ],
   'actuator':[
       {'pos_range':(0, 1), 'vel_range':(-2, 4), 'hdr_id':0, 'name':'robotiq_2f_85', 'scale':-0.08, 'offset':0.08},
   ],
   'cam': []
},

# device3: interface, sensors, actuators, cams configs
'right_cam':{
   'interface': {'type': 'realsense', 'rgb_topic':'image_rgb', 'd_topic':'image_d'},
   'sensor':[],
   'actuator':[]
   'cam': [
       {'range':(0, 255), 'noise':0.00, 'hdr_id':'rgb', 'scale':1, 'offset':0, 'name':'/color/image_raw'},
       {'range':(0, 255), 'noise':0.00, 'hdr_id':'d', 'scale':1, 'offset':0, 'name':'/depth_uncolored/image_raw'},
   ],
},
```


# Configs
Robot class is configured using a config file. The config file is essentially a nested python dictionary and contains details about the various robotics components (`device`) that come together to define a robot. For each robotic device following details can be provided:

- **`name`**: Symbolic name (for easy reference) of robotic device being configured.

- **`interface`**: Specifies the connection configurations of the robotic device -
   - `type`: Type of the hardware being configured. This has to be one of the supported hardwares. A list of all supported hardware can be found [here](#hardware-robot).
   - `...`:  any additional arguments required by the connection. These additional arguments are directly passed to the constructor of respective hardware's class (`robohive/robot/hardware_base:hardwareBase`).

- **`sensor`**: An ordered list of all available sensors. List ordering must follow the same order as specified in the MuJoCo model. List ordering is used to implicitly determine sensor's index in the sensor array -
   - `name`: Name of the sensor. Note: This has to be the same as the sensor name in the MuJoCo model.
   - `range`: Range of values expected from this sensor. Readings outside these ranges are clamped to the limits.
   - `noise`: expected amplitude of noise in the sensor readings. Use this parameter to add noise to the simulated sensor reading. Noise is sampled from a uniform distribution between `(-noise, +noise)` and added to the sensor reading. It has no effect if the robot is instantiated using hardware backend. ```sensor += noise_scale*sensor['noise']*self.np_random.uniform(low=-1.0, high=1.0)``` where `noise_scale` is a parameter (of the entire robot) to scale the noise of the whole robot.
   - `hdr_id`: Hardware index of the sensor in the hardware's sensor array.
   - `scale`/`offset`: parameters to map hardware sensor values to simulated sensor values `sensor_sim = sensor_hdr*scale+offset`

   **Note**: Currently only scalar sensors are supported. Multi-valued sensor support is on the wishlist.

- **`cams`**: cams are special cases of sensors that provide multi-dimensional readings. 3D for RGB and 2D for depth.
   - `name`: Name of the sensor. Since there are no explicit camera sensors in MuJoCo, there are no specific constraints on camera names. We internally use the pub-sub topics names as camera names.
   - `range`: see description in sensor
   - `noise`:see description in sensor
   - `hdr_id`: see description in sensor
   - `scale`/`offset`: see description in sensor

- **`actuator`**: An ordered list of all available actuators. List ordering must follow the same order as specified in the MuJoCo model. List ordering is used to implicitly determine the actuator's index in the actuator array -
   - `name`: Name of the actuator. Note: This has to be the same as the actuator name in the MuJoCo model.
   - `range`: Range of values the actuator can accept in the sim space. Readings outside these ranges are clamped to the limits.
   - `noise`: [_Wishlist_] expected amplitude of noise in actuator's performance. Use this parameter to add noise to the simulated actuator demands. Noise is sampled from a uniform distribution between `(-noise, +noise)` and added to the actuator demands. It has no effect if the robot is instantiated using hardware backend. ```ctrl_sim += noise_scale*act['noise']*self.np_random.uniform(low=-1.0, high=1.0)``` where `noise_scale` is a parameter (of the entire robot) to scale the noise of the whole robot.
   - `hdr_id`: Hardware index of the actuator in the hardware's sensor array.
   - `scale`/`offset`: parameters to map simulated desired values to hardware actuator demands `ctrl_hdr = ctrl_sim*scale+offset`


RoboHive [robot](robot.py) class consumes the simulation and the configurations files to provide a unified class that can be used as a general robot abstraction -- either inside a MDP (OpenAI-gym, DM-Control) formulation, or to directly talk to the robot ([tutorial](../tutorials/examine_robot.py)).