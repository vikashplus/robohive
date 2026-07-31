""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import inspect
import os
import time
from collections import deque

import mujoco
import numpy as np

from robohive.physics.sim_scene import SimScene
from robohive.robot.hardware_base import HARDWARE_REGISTRY, SENSOR_POSTPROCESS
from robohive.utils.prompt_utils import Prompt, prompt

np.set_printoptions(precision=4)


_ROBOT_VIZ = False

# TODO ===========================================
# rename robot_config something more meaningful
# support loading multiple config files
# seperate ROBOT_VIZ as its own class
# remap_space() needs rigerous testing
# Support for sensors that provide multiple reading values. Sensor indexing might not directly follow the sensor's list index in this case. This support will potentilly allow us to also list cams as sensors
# Support for non uniform noise in sensor readings
# Support for noisy actions + separate noise_scale for sensor and actuator
# rename pos/vel to act/delta_act

# NOTE/ GOOD PRACTICES ===========================
# nq should be nv
# Order of sensors and actuators in config should follow XML order
# Space definitions
    # sim_id: ID (defined by order in mujoco's xml) of the sensor/actuator in the sim
    # hdr_id: ID (defined by the order in .config) of the sensor/actuator in the robot_config (hardware) space (robot_config unifies different hardware into a single unified hardware space)
    # hdr_adr: Physical address of the sensor/actuator (defined by hardware specs) (e.g. dynamixel - This is the address/motor_id used to communicate with the individual motors)



class Robot():
    """
    A unified viewpoint of robot between simulation(sim) and hardware(hdr)
    """
    # Cached a persistent connection to the robot that is shared for the application's lifetime.
    robot_config = None

    def __init__(self,
                robot_name:str = 'default_robot',
                model_path: str = None,     # model file to create sim
                mj_sim = None,              # pass sim directly
                config_path: str = None,    # config defining sensors and actuator groups
                act_mode: str = "pos",      # pos / vel
                is_hardware:bool = None,    # use hardware
                sensor_cache_maxsize = 5,   # cache size for sensors
                noise_scale = 0,            # scale for sensor noise
                random_generator = None,    # random number generator
                **kwargs,
            ):

        if kwargs != {}:
            prompt("Warning: Unused kwargs found: {}".format(kwargs), type=Prompt.WARN)
        self.name = robot_name+'(sim)' if is_hardware is None else robot_name+'(hdr)'
        self._act_mode = act_mode
        self.is_hardware = bool(is_hardware)
        self._sensor_cache_maxsize = sensor_cache_maxsize
        self._noise_scale = noise_scale
        if random_generator == None:
            self.np_random = np.random
        else:
            self.np_random = random_generator

        # sensor cache
        self._sensor_cache = deque([], maxlen=self._sensor_cache_maxsize)

        # create robot sim
        if mj_sim is None:
            # (creates new robot everytime to facilitate parallelization)
            prompt("Preparing robot-sim from %s" % model_path)
            self.sim =SimScene.get_sim(model_handle=model_path)
        else:
            # use provided sim
            self.sim = mj_sim

        # Configure the robot
        if self.robot_config is None:
            prompt("Configuring a new session for {}".format(self.name), 'white', 'on_grey')
            self.robot_config = self.configure_robot(self.sim, config_path)
            if _ROBOT_VIZ:
                self.configure_robot_viz(self.robot_config)
            # start the robot
            if self.is_hardware is True:
                prompt("Initializing robot: %s"%(self.name), 'white', 'on_grey')
                self.robot_config = self.hardware_init(self.robot_config)
        else:
            prompt("Reusing a previours session of {}".format(self.name), 'white', 'on_grey')

        # check robot health
        if self.is_hardware is True:
            self.hardware_okay(self.robot_config)

            # disable all collisions
            self.sim.model.geom_conaffinity[:] = 0
            self.sim.model.geom_contype[:] = 0

        # Robot's time
        self.time_start = time.time()
        self.time_wall = time.time()-self.time_start # Wall time (used for realtime factors) for both sim and hardware

        # refresh the sensor cache
        self._sensor_cache_refresh()


    # Check if all hardware components are okay
    def hardware_okay(self, robot_config):
        for name, device in robot_config.items():
            if not device['robot'].okay():
                prompt("ERROR: Please check device {}".format(name), 'white', 'on_red')


    # initialize all hardware components
    def hardware_init(self, robot_config):

        # initalize
        for name, device in robot_config.items():
            prompt("Initializing device: %s"%(name), 'white', 'on_grey')
            itype = device['interface']['type']
            cls = HARDWARE_REGISTRY.get(itype)
            if cls is None:
                raise NotImplementedError(
                    "No hardwareBase registered for interface.type={!r}. Registered types: {}. "
                    "Import the module that registers this type before constructing "
                    "Robot(is_hardware=True).".format(itype, sorted(HARDWARE_REGISTRY)))
            iface = {k: v for k, v in device['interface'].items() if k != 'type'}
            # Most hardware classes are self-contained (ip/port/etc. from `interface` is
            # enough). A few (dynamixel-bus devices, which manage several motors sharing
            # one connection and need per-motor address/mode bookkeeping) opt in to
            # receiving the full robot_config device dict by naming `device` as a
            # constructor parameter; everyone else never sees it.
            if 'device' in inspect.signature(cls).parameters:
                iface['device'] = device
            device['robot'] = cls(name=name, **iface)

        # start all hardware
        for name, device in robot_config.items():
            device['robot'].connect()

        return robot_config


    # get hardware sensors
    def hardware_get_sensors(self):
        current_sensor_value = {}
        current_sensor_value['time'] = time.time() - self.time_start
        for name, device in self.robot_config.items():
            if 'sensor' in device.keys() and len(device['sensor'])>0:
                itype = device['interface']['type']
                raw = device['robot'].get_sensors()
                postprocess = SENSOR_POSTPROCESS.get(itype, lambda raw: raw['pos'])
                vals = postprocess(raw)

                # calibrate sensors. vals is positionally ordered to match device['sensor']
                # (both built by iterating the same config-declared list, see configure_robot()).
                arr = np.empty(len(device['sensor']), dtype=np.float64)
                for id, sensor in enumerate(device['sensor']):
                    arr[id] = vals[id]*sensor['scale'] + sensor['offset']
                current_sensor_value[name] = arr
                device['sensor_data'] = arr
                device['sensor_time'] = current_sensor_value['time']
        return current_sensor_value


    # apply controls to hardware
    def hardware_apply_controls(self, control, space='hdr'):
        """
        Send one dt's worth of (already-clipped, locally-achievable) controls to hardware.
        control: control vector in hdr or sim space
        space: 'hdr' or 'sim' (defaults to 'hdr' as represented in robot_config)
        """

        for name, device in self.robot_config.items():
            if 'actuator' in device.keys() and len(device['actuator'])>0:
                # hw_q is positionally ordered to match device['actuator'] (both built by
                # iterating the same config-declared list, see configure_robot()).
                hw_q = []
                for actuator in device['actuator']:
                    ctrl = control[actuator['sim_id']] if space == 'sim' else control[actuator['hdr_id']]
                    hw_q.append(ctrl*actuator['scale'] + actuator['offset'])
                device['robot'].apply_commands(hw_q)


    # move actuated dofs to a target position (blocking, large-displacement — distinct from
    # the per-dt hardware_apply_controls above; hardware classes implement this via their own
    # min-jerk/via-point trajectories)
    def hardware_reset(self, reset_pos, space='sim'):
        """
        reset_pos: target qpos vector
        space: 'sim' (full sim qpos vector, addressed via data_id) or
               'hdr' (per-actuator vector ordered by hdr_id, as in hardware_apply_controls)
        """
        assert space in ['sim', 'hdr'], "space must be 'sim' or 'hdr'"
        for name, device in self.robot_config.items():
            if name == 'default_robot':
                continue
            qpos_actuators = [a for a in device.get('actuator', []) if a['data_type'] == 'qpos']
            if qpos_actuators:
                hw_q = []
                for a in qpos_actuators:
                    pos = reset_pos[a['data_id']] if space == 'sim' else reset_pos[a['hdr_id']]
                    pos = np.clip(pos, a['pos_range'][0], a['pos_range'][1])
                    hw_q.append(pos*a['scale'] + a['offset'])
                device['robot'].reset(hw_q)
            else:
                # passive device (tendon-driven gripper, camera, etc.) — no qpos target to
                # compute; let the device bring itself to its own known reset state
                device['robot'].reset()

    # close hardware
    def hardware_close(self):
        status = True
        for name, device in self.robot_config.items():
            if device.get('robot'):
                print("Closing {} connection".format(device['interface']['type']))
                status = device['robot'].close()
                if status is True:
                    device['robot'] = None
        return status


    # configure robot
    def configure_robot(self, sim, config_path):
        """
        Read the model xml and robot configs from provided files. Compile config with the model
        """
        if config_path is None:
            robot_config = {}
            robot_config['default_robot'] = {'sensor': ['qpos', 'qvel', 'act'], 'actuator': 'actuator'}
            return robot_config

        prompt("Reading robot-configurations from %s" % config_path)
        with open(config_path, 'r') as f:
            robot_config = eval(f.read())

        hdr_sensor_id = -1 # IDs as per the config order
        hdr_actuator_id = -1 # IDs as per the config order
        for name, device in robot_config.items():
            prompt("Configuring component %s"% name)

            # configure device sensors
            device['sensor_ids'] = []
            device['sensor_names'] = []
            for sensor in device['sensor']:
                hdr_sensor_id += 1
                sensor['hdr_id'] = hdr_sensor_id
                sensor['sim_id'] = sim.model.sensor_name2id(sensor['name'])
                device['sensor_names'].append(sensor['name']) # list of all ids
                device['sensor_ids'].append(sensor['hdr_adr']) # list of all ids
                sensor_type = sim.model.sensor_type[sensor['sim_id']]
                sensor_objid = sim.model.sensor_objid[sensor['sim_id']]
                # sensordata_id: address in sim.data.sensordata for this sensor.
                # Stored for all sensor types so Pass 2 of sensor2sim can write
                # hardware readings into sensordata as the single authoritative source.
                sensor['sensordata_id'] = int(sim.model.sensor_adr[sensor['sim_id']])
                if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:  # scalar joint position (hinge and slide only)
                    sensor['data_type'] = 'qpos'
                    sensor['data_id'] = sim.model.jnt_qposadr[sensor_objid]
                elif sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL:  # scalar joint velocity (hinge and slide only)
                    sensor['data_type'] = 'qvel'
                    sensor['data_id'] = sim.model.jnt_dofadr[sensor_objid]
                else:
                    # Non-invertible sensor: sim.forward() recomputes this from state.
                    # Covers tendon position, joint actuator force, actuator force, etc.
                    sensor['data_type'] = 'sensordata'
                    sensor['data_id'] = sensor['sensordata_id']

            # configure device actuators
            device['actuator_ids'] = []
            device['actuator_names'] = []
            for actuator in device['actuator']:
                hdr_actuator_id += 1
                actuator['hdr_id'] = hdr_actuator_id
                actuator['sim_id'] = sim.model.actuator_name2id(actuator['name'])
                device['actuator_names'].append(actuator['name']) # list of all ids
                device['actuator_ids'].append(actuator['hdr_adr']) # list of all ids
                actuator_trntype = sim.model.actuator_trntype[actuator['sim_id']]
                actuator_trnid = sim.model.actuator_trnid[actuator['sim_id'], 0]
                if actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT:  # // force on joint
                    actuator['data_type'] = 'qpos'
                    # qpos is indexed by jnt_qposadr, not jnt_dofadr; the two
                    # diverge when a free/ball joint precedes this joint
                    # (freejoint: 7 qpos vs 6 dofs), shifting every read.
                    actuator['data_id'] = sim.model.jnt_qposadr[actuator_trnid]
                elif actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON:  # force on tendon
                    actuator['data_type'] = 'ten_length'
                    actuator['data_id'] = actuator_trnid
                else:
                    quit("ERROR: actuator {} has unsupported transmission_type: {}".format(actuator['name'],actuator_trntype))
        return robot_config


    # refresh the sensor cache
    def _sensor_cache_refresh(self):
        for _ in range(self._sensor_cache_maxsize):
            self.get_sensors()


    # get past sensor
    def get_sensor_from_cache(self, index=-1):
        assert (index>=0 and index<self._sensor_cache_maxsize) or \
                (index<0 and index>=-self._sensor_cache_maxsize), \
                "cache index out of bound. (cache size is %2d)"%self._sensor_cache_maxsize
        return self._sensor_cache[index]


    # get sensor data and update robot time accordingly
    def get_sensors(self, noise_scale=None, random_generator=None):
        """
        Get sensor data
        """
        current_sen={}
        noise_scale = self._noise_scale if noise_scale is None else noise_scale

        # Update time
        self.time_wall = time.time()-self.time_start

        if self.is_hardware:
            # record sensor*device['scale']+device['offset']
            current_sen = self.hardware_get_sensors()
            # update the sim as per the hardware observations
            self.sensor2sim(current_sen, self.sim)
        else:
            current_sen['time']= self.sim.data.time # data time stamp
            for name, device in self.robot_config.items():
                if name == "default_robot":
                    sen = {}
                    sen['qpos'] = self.sim.data.qpos.copy()
                    sen['qvel'] = self.sim.data.qvel.copy()
                    sen['act'] = self.sim.data.act.copy() if self.sim.model.na >0 else None
                    current_sen[name] = sen
                else:
                    sen = []
                    for sensor in device['sensor']:
                        s = self.sim.data.sensordata[sensor['sim_id']]
                        # ensure range
                        s = np.clip(s, sensor['range'][0], sensor['range'][1])
                        # add noise
                        if noise_scale!=0:
                            s += noise_scale*sensor['noise']*self.np_random.uniform(low=-1.0, high=1.0)
                        # ensure range
                        s = np.clip(s, sensor['range'][0], sensor['range'][1])
                        sen.append(s)
                    current_sen[name] = np.array(sen)

                # create sensor reading
                device['sensor_data'] = current_sen[name]
                device['sensor_time'] = current_sen['time']

            # VIK???: Propagating sensors back to sim can create trouble with contact stability in presence of noise
            # self.sensor2sim(current_sen, self.sim)

        # cache sensors
        self._sensor_cache.append(current_sen)

        return current_sen


    # get sensor data and update robot time accordingly
    def get_visual_sensors(self, height:int, width:int, cameras:list, device_id:int, sim):

        if self.is_hardware:
            imgs = np.zeros((len(cameras), height, width, 3), dtype=np.uint8)
            depths = np.zeros((len(cameras), height, width), dtype=np.uint16)

            current_sensor_value = {}
            current_sensor_value['time'] = time.time() - self.time_start

            for ind, cam_name in enumerate(cameras):
                assert cam_name in self.robot_config.keys(), "{} camera not found".format(cam_name)
                device = self.robot_config[cam_name]

                if hasattr(device['robot'], 'get_frame'):
                    # RGB-only camera (e.g. a UVC/webcam) — no depth stream.
                    rgb = device['robot'].get_frame()
                    data = {'time': time.time() - self.time_start, 'rgb': rgb, 'd': None}
                else:
                    # RealSense-style camera: get_sensors() itself returns {'rgb','d'}.
                    data = device['robot'].get_sensors()
                data_height = data['rgb'].shape[0]
                assert data_height == height, "Incorrect image height: required:{}, found:{}".format(height, data_height)
                data_width = data['rgb'].shape[1]
                assert data_width == width, "Incorrect image width: required:{}, found:{}".format(width, data_width)
                current_sensor_value[cam_name] = data

                # calibrate sensors
                for cam in device['cam']:
                    current_sensor_value[cam_name][cam['hdr_adr']] = current_sensor_value[cam_name][cam['hdr_adr']]*cam['scale'] + cam['offset']
                device['sensor_data'] = current_sensor_value[cam_name]
                device['sensor_time'] = current_sensor_value['time']
                imgs[ind, :, :, :] = current_sensor_value[cam_name]['rgb']
                if current_sensor_value[cam_name]['d'] is not None:
                    depths[ind, :, :] = current_sensor_value[cam_name]['d'][:,:,0] # assumes single channel depth

        else:
            imgs = np.zeros((len(cameras), height, width, 3), dtype=np.uint8)
            depths = np.zeros((len(cameras), height, width))
            for ind, cam in enumerate(cameras):
                # img, depth = sim.render(width=width, height=height, depth=True, mode='offscreen', camera_name=cam, device_id=device_id)
                img, depth = sim.renderer.render_offscreen(width=width, height=height, depth=True, camera_id=cam, device_id=device_id)
                # img = img[::-1, :, :] # Image has to be flipped
                imgs[ind, :, :, :] = img
                depths[ind, :, :] = depth

        return imgs, depths


    # Propagate sensor values back through the sim.
    def sensor2sim(self, sensor, sim):
        """
        Propagate sensor values back through the sim.
        This operation is water-tight only where the system is fully observable (including velocities)
        Example usage:
          (1) can be used to feed hardware sensors to robot-sim
          (2) can be used to feed noisy sim sensors back into the robot-sim. Note: Be careful, sim might not be stable for simulation after
        """
        if not self.is_hardware and (self._noise_scale!=0):
            print("WARNING: Propagating noisy sensors back to sim can destablize simulation")

        sim.data.time = sensor['time']

        # Pass 1: write invertible state — qpos and qvel survive sim.forward()
        for name, device in self.robot_config.items():
            if name == "default_robot":
                sim.data.qpos[:] = device['sensor_data']['qpos']
                sim.data.qvel[:] = device['sensor_data']['qvel']
                if self.sim.model.na > 0:
                    sim.data.act[:] = device['sensor_data']['act']
            else:
                for s_id, s_val in enumerate(device['sensor']):
                    if s_val['data_type'] in ('qpos', 'qvel'):
                        getattr(sim.data, s_val['data_type'])[s_val['data_id']] = sensor[name][s_id]

        # Propagate qpos/qvel through kinematics; also recomputes sensordata from sim physics
        sim.forward()

        # Pass 2: write all hardware readings into sensordata, making it the single
        # authoritative source regardless of sensor type.  For qpos/qvel sensors this
        # is redundant (forward() already propagated them) but keeps sensordata complete.
        for name, device in self.robot_config.items():
            if name == "default_robot":
                continue
            for s_id, s_val in enumerate(device['sensor']):
                sim.data.sensordata[s_val['sensordata_id']] = sensor[name][s_id]


    # synchronize states between two sims
    def sync_sims(self, source_sim, destination_sim, model=True, data=True):
        destination_sim.data.time = source_sim.data.time
        if data:
            destination_sim.data.qpos[:] = source_sim.data.qpos[:].copy()
            destination_sim.data.qvel[:] = source_sim.data.qvel[:].copy()
            if destination_sim.model.na>0:
                destination_sim.data.act[:] = source_sim.data.act[:].copy()
            if destination_sim.model.nmocap>0:
                destination_sim.data.mocap_pos[:] = source_sim.data.mocap_pos.copy()
                destination_sim.data.mocap_quat[:] = source_sim.data.mocap_quat.copy()

        if model:
            if destination_sim.model.nsite>0:
                destination_sim.model.site_pos[:] = source_sim.model.site_pos[:].copy()
                destination_sim.model.site_quat[:] = source_sim.model.site_quat[:].copy()
            if destination_sim.model.nbody>0:
                destination_sim.model.body_pos[:] = source_sim.model.body_pos[:].copy()
                destination_sim.model.body_quat[:] = source_sim.model.body_quat[:].copy()

        destination_sim.forward()


    # remap sensor/actuators spaces: sim<>hardware, TODO: Needs rigerous testing
    def remap_space(self, input_vec, input_type:str, input_space:str, output_space:str):
        assert input_type in ['sensor', 'actuator'], "check input type"
        assert input_space in ['sim', 'hdr'], "check input space"
        assert output_space in ['sim', 'hdr'], "check output space"
        assert input_space != output_space, "Check: Input and output spaces are the same"

        input_space = input_space+'_id'
        output_space = output_space+'_id'
        output_vec = input_vec.copy()

        # sim => hdr
        if input_space == 'sim_id' and output_space == 'hdr_id':
            output_space == 'data_id'# WARNING: This is a hack as we don't have physical/logical id
            for name, device in self.robot_config.items():
                if input_type == 'actuator' and 'actuator' in device.keys() and len(device['actuator'])>0:
                    for id, actuator in enumerate(device['actuator']):
                        output_vec[actuator[output_space]] = input_vec[actuator[input_space]]*actuator['scale'] + actuator['offset']

                if input_type == 'sensor' and 'sensor' in device.keys() and len(device['sensor'])>0:
                    for id, sensor in enumerate(device['sensor']):
                        output_vec[sensor[output_space]] = (input_vec[sensor[input_space]] - sensor['offset'])/sensor['scale']

        # hdr => sim
        if input_space == 'hdr_id' and output_space == 'sim_id':
            input_space == 'data_id' # WARNING: This is a hack as we don't have physical/logical id
            for name, device in self.robot_config.items():
                if input_type == 'actuator' and 'actuator' in device.keys() and len(device['actuator'])>0:
                    for id, actuator in enumerate(device['actuator']):
                        output_vec[actuator[output_space]] = (input_vec[actuator[input_space]] - actuator['offset'])/actuator['scale']

                if input_type == 'sensor' and 'sensor' in device.keys() and len(device['sensor'])>0:
                    for id, sensor in enumerate(device['sensor']):
                        output_vec[sensor[output_space]] = input_vec[sensor[input_space]]*sensor['scale'] + sensor['offset']
        return output_vec


    # Normalize actions from absolute space to unit space
    def normalize_actions(self, controls, out_space='sim', unnormalize=False):
        """
        Normalize actions from absolute space to unit space
        Recover actions from unit space to absolute space; if unnormalize==True
        in_space for controls has to be 'sim'
        """
        controls_out = controls.copy()
        for name, device in self.robot_config.items():
            if name == "default_robot":
                if self._act_mode == "pos":
                    act_mid = np.mean(self.sim.model.actuator_ctrlrange, axis=-1)
                    act_rng = (self.sim.model.actuator_ctrlrange[:,1]-self.sim.model.actuator_ctrlrange[:,0])/2.0
                    controls_out = controls*act_rng+act_mid if unnormalize else (controls-act_mid)/act_rng
                else:
                    raise TypeError("only pos act supported")
            else:
                for actuator in device['actuator']:
                    in_id = actuator['sim_id']
                    # output ordering is as per the config order for hdr
                    out_id = actuator['sim_id'] if out_space == 'sim' else actuator['hdr_id']

                    if self._act_mode == "pos":
                        act_mid = (actuator['pos_range'][1]+actuator['pos_range'][0])/2.0
                        act_rng = (actuator['pos_range'][1]-actuator['pos_range'][0])/2.0
                    elif self._act_mode == "vel":
                        act_mid = (actuator['vel_range'][1]+actuator['vel_range'][0])/2.0
                        act_rng = (actuator['vel_range'][1]-actuator['pos_range'][0])/2.0
                    else:
                        raise TypeError("Unknown act mode: {}".format(self._act_mode))

                    # unnormalize/ normalize
                    control = controls[in_id]
                    if unnormalize:
                        control = np.clip(control, -1, 1)
                        control = control*act_rng+act_mid
                    else:
                        control =  (control-act_mid)/act_rng
                        control = np.clip(control, -1, 1)

                    # remap to desired space
                    controls_out[out_id] = control
        return controls_out


    # enfoce limits
    def process_actuator(self, controls, step_duration,
            normalized=True,
            position_limits=True,
            velocity_limits=True,
            out_space='sim'):
        """
        Process the actuation demands to
            (1) Remap provided controls to actuation space,
            (2) Enforces hardware position and velocity limits on the controls
        """
        # last_obs = self.get_sensor_from_cache(-1)
        processed_controls = controls.copy()
        for name, device in self.robot_config.items():
            if name == "default_robot":
                if self._act_mode == "pos":
                    if normalized:
                        processed_controls = np.mean(self.sim.model.actuator_ctrlrange, axis=-1)+ \
                            controls*(self.sim.model.actuator_ctrlrange[:,1]-self.sim.model.actuator_ctrlrange[:,0])/2.0
                else:
                    raise TypeError("only pos act supported")
            else:
                for actuator in device['actuator']:
                    in_id = actuator['sim_id']
                    # output ordering is as per the config order for hdr
                    out_id = actuator['sim_id'] if out_space == 'sim' else actuator['hdr_id']

                    control = controls[in_id]
                    if self._act_mode == "pos":
                        # remap to the limits if normalized
                        if normalized:
                            control = (actuator['pos_range'][1]+actuator['pos_range'][0])/2.0 + \
                                        control*(actuator['pos_range'][1]-actuator['pos_range'][0])/2.0
                        # enforce velocity limits
                        # ALERT: This depends on previous sensor. This is not ideal as it breaks MDP addumptions. Be careful
                        if velocity_limits:
                            last_obs = getattr(self.sim.data, actuator["data_type"])[actuator["data_id"]]
                            ctrl_desired_vel = (control - last_obs)/step_duration
                            ctrl_feasible_vel = np.clip(ctrl_desired_vel, actuator['vel_range'][0], actuator['vel_range'][1])
                            control = last_obs + ctrl_feasible_vel*step_duration
                    elif self._act_mode == "vel":
                        # remap to the limits if normalized
                        if normalized:
                            control = (actuator['vel_range'][1]+actuator['vel_range'][0])/2.0 + \
                                        control*(actuator['vel_range'][1]-actuator['vel_range'][0])/2.0
                        # enforce velocity limits
                        # ALERT: This depends on previous sensor. This is not ideal as it breaks MDP addumptions. Be careful
                        last_obs = getattr(self.sim.data, actuator["data_type"])[actuator["data_id"]]
                        control = last_obs + control*step_duration
                    else:
                        raise TypeError("Unknown act mode: {}".format(self._act_mode))

                    # enforce position limits
                    if position_limits:
                        control = np.clip(control, actuator['pos_range'][0], actuator['pos_range'][1])

                    # remap to desired space
                    processed_controls[out_id] = control

        return processed_controls


    # step the robot one step forward in time
    def step(self, ctrl_desired, step_duration, ctrl_normalized=True, realTimeSim=False, render_cbk=None):
        """
        Apply controls and step forward in time
        INPUTS:
            ctrl_desired:       Desired control to be applied(sim_space)
            step_duration:      Step duration (seconds)
            ctrl_normalized:    is the ctrl normalized to [-1, 1]
            realTimeSim:        run simulate real world speed via sim
        """

        # pick output space
        robot_type = 'hdr' if self.is_hardware else 'sim'

        # enforce limits
        ctrl_feasible = self.process_actuator(controls=ctrl_desired, step_duration=step_duration,\
             normalized=ctrl_normalized, position_limits=True, velocity_limits=True, out_space=robot_type)

        # Send controls to the robot
        if self.is_hardware:
            self.hardware_apply_controls(ctrl_feasible, space=robot_type)
            if render_cbk:
                render_cbk()
        else:
            n_frames=int(step_duration/self.sim.step_duration)
            self.sim.data.ctrl[:] = ctrl_feasible
            self.sim.advance(substeps=n_frames, render=(render_cbk!=None))

        # update viz
        if _ROBOT_VIZ:
            for name, device in self.robot_config.items():
                device['controls'] = []
                for actuator in device['actuator']:
                    device['controls'].append(ctrl_feasible[actuator['sim_id']])
            self.update_robot_viz(update_sensor=True, update_control=True)

        # synchronize time to maintain step_duration
        if self.is_hardware or realTimeSim:
            time_now = (time.time() - self.time_start)
            time_left_in_step = step_duration - (time_now-self.time_wall)
            if (time_left_in_step > 0.001):
                time.sleep(time_left_in_step)
            elif time_left_in_step < 0.0:
                prompt("Step duration %0.4fs, Step took %0.4fs, Time left %0.4f"% (step_duration, (time_now-self.time_wall), time_left_in_step), type=Prompt.WARN)

        if _ROBOT_VIZ:
            global timing_SRV_t0
            timing_SRV_t = time.time()
            timing_SRV.append(y_data=timing_SRV_t-timing_SRV_t0)
            timing_SRV_t0 = timing_SRV_t
        return ctrl_feasible


    # Reset the robot
    def reset(self,
              reset_pos,
              reset_vel,
              blocking = True,
              **kwargs
              ):

        prompt("Resetting {}".format(self.name), 'white', 'on_grey', flush=True)

        # Enforce specs on the request
        #   for actuated dofs => actuator specs
        #   for passive dofs => sensor specs
        feasibe_pos = reset_pos.copy()
        feasibe_vel = reset_vel.copy()
        for name, device in self.robot_config.items():
            if name != "default_robot":
                if len(device['actuator'])>0: # actuated dofs
                    for actuator in device['actuator']:
                        if actuator['data_type'] == 'qpos':
                            feasibe_pos[actuator['data_id']] = np.clip(reset_pos[actuator['data_id']], actuator['pos_range'][0], actuator['pos_range'][1])
                else: # passive dofs
                    for sensor in device['sensor']:
                        if sensor['data_type'] == 'qpos':
                            feasibe_pos[sensor['data_id']] = np.clip(reset_pos[sensor['data_id']], sensor['range'][0], sensor['range'][1])
                        elif sensor['data_type'] == 'qvel':
                            feasibe_vel[sensor['data_id']] = np.clip(reset_vel[sensor['data_id']], sensor['range'][0], sensor['range'][1])

        if self.is_hardware:
            t_reset_start = time.time()
            prompt("\nRollout took:{}".format(t_reset_start- self.time_start))
            prompt("\aResetting {}: ".format(self.name), 'white', 'on_grey', flush=True, end="")
            # send request to all devices, actuated and passive alike
            # feasibe_pos was built above via actuator['data_id'] indexing, i.e. sim space
            self.hardware_reset(feasibe_pos, space='sim')

            if blocking:
                input("press a key to start rollout")
            prompt(" Done in {}".format(time.time()-t_reset_start), 'white', 'on_grey', flush=True)
        else:
            # Ideally we should use actuator/ reset mechanism as in the real world
            # but choosing to directly resetting sim for efficiency
            self.sim.reset()
            self.sim.data.qpos[:] = feasibe_pos
            self.sim.data.qvel[:] = feasibe_vel
            self.sim.forward() # ???Vik alternatively should following be called? functions.mj_step1(self.sim.model, self.sim.data)

            if _ROBOT_VIZ:
                input("press a key to start rollout")

        # clear viz
        if _ROBOT_VIZ:
            self.clear_robot_viz(clear_sensor=True, clear_control=True)

        # refresh sensor cache before exiting reset
        self._sensor_cache_refresh()

        # restart the robot clock
        self.time_start = time.time()
        self.time_wall = time.time()-self.time_start

        global timing_SRV_t0
        timing_SRV_t0 = time.time()

        return feasibe_pos, feasibe_vel


    # Clear the robot class. Note that it doesn't close the persistent connection
    def __del__(self):
        if self.robot_config is not None and self.is_hardware:
            raise RuntimeWarning("RoboHive:> Robot class is being cleared from the workspace. This is expected if we still need to maintain the active connection to the hardware. A persistent connection to robot is still maintained and will be used next time a robot class is created. Ensure that a robot.close() is called to terminate the persistent connection before exiting the program.")

    # Close the persistnent connection to the robot. This should be called only once at the end when persistent connection is no longer needed.
    def close(self):
        if self.robot_config is not None:
            status = self.hardware_close() if self.is_hardware else True
            if status:
                prompt(f"Closed {self.name} (Status: {status})", 'white', 'on_grey', flush=True)
                self.robot_config = None
            else:
                prompt(f"Error closing {self.name} (Status: {status})", 'red', 'on_grey', flush=True, type=Prompt.ERROR)
        else:
            prompt(f"Trying to close a non-existent robot", flush=True, type=Prompt.WARN)


def demo_robot():
    from robohive.utils import gym

    prompt("Starting Robot===================")
    envw = gym.make('FrankaReachFixed-v0')
    env = envw.unwrapped
    rob = env.robot

    prompt("Getting sensor data==============")
    sen = rob.get_sensors()
    prompt("Sensor data: ", end="")
    prompt(sen)

    prompt("stepping forward=================")
    ctrl = env.np_random.uniform(size=env.sim.model.nu)
    rob.step(ctrl, 1.0)

    prompt("Resetting Robot==================")
    pos = env.np_random.uniform(size=env.sim.model.nq)
    vel = env.np_random.uniform(size=env.sim.model.nv)
    rob.reset(pos, vel)

    prompt("Closing Robot====================")
    rob.close()

if __name__ == '__main__':
    demo_robot()
