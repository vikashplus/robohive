""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com), Sudeep Dasari (sdasari@andrew.cmu.edu )
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
from mj_envs.envs import env_base
from mj_envs.envs.tcdm.reference_motion import ReferenceMotion
from mj_envs.utils.quat_math import quat2euler
import numpy as np
import os
import collections

# from pyquaternion import Quaternion

# ToDo
# - change target to reference


class TrackEnv(env_base.MujocoEnv):

    DEFAULT_CREDIT = """\
    Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps
        Sudeep Dasari, Abhinav Gupta, Vikash Kumar
        ICRA-2023 | https://pregrasps.github.io
    """

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'robot_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }
    def __init__(self, object_name, model_path, obsd_model_path=None, seed=None, **kwargs):

        print(self.DEFAULT_CREDIT)
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Process model_path to import the right object
        with open(curr_dir+model_path, 'r') as file:
            processed_xml = file.read()
            processed_xml = processed_xml.replace('OBJECT_NAME', object_name)
        processed_model_path = curr_dir+model_path[:-4]+"_processed.xml"
        with open(processed_model_path, 'w') as file:
            file.write(processed_xml)
        self._object_name = object_name
        # Process obsd_model_path to import the right object
        if obsd_model_path == model_path:
            processed_obsd_model_path = processed_model_path
        elif obsd_model_path:
            with open(curr_dir+obsd_model_path, 'r') as file:
                processed_xml = file.read()
                processed_xml = processed_xml.replace('OBJECT_NAME', object_name)
            processed_obsd_model_path = curr_dir+model_path[:-4]+"_processed.xml"
            with open(processed_obsd_model_path, 'w') as file:
                file.write(processed_xml)
        else:
            processed_obsd_model_path = None

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, object_name, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=processed_model_path, obsd_model_path=processed_obsd_model_path, seed=seed)
        os.remove(processed_model_path)
        if processed_obsd_model_path and processed_obsd_model_path!=processed_model_path:
            os.remove(processed_obsd_model_path)

        self._setup(**kwargs)


    def _setup(self,
               reference,                       # reference target/motion for behaviors
               motion_start_time:float=0,       # useful to skip initial motion
               motion_extrapolation:bool=True,  # Hold the last frame if motion is over
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):

        # prep reference
        self.ref = ReferenceMotion(reference=reference, motion_extrapolation=motion_extrapolation)
        self.motion_start_time = motion_start_time
        self.target_sid = self.sim.model.site_name2id("target")

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=10,
                       **kwargs)
        # Adjust horizon if not motion_extrapolation
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.ref.horizon # doesn't work always. WIP
        # Adjust init as per the specified key
        if self.sim.model.nkey>0:
            self.init_qpos[:] = self.sim.model.key_qpos[0,:]


    # def to_quat(self, arr):
    #     if isinstance(arr, Quaternion):
    #         return arr.unit
    #     if len(arr.shape) == 2:
    #         return Quaternion(matrix=arr).unit
    #     elif len(arr.shape) == 1 and arr.shape[0] == 9:
    #         return Quaternion(matrix=arr.reshape((3,3))).unit
    #     return Quaternion(array=arr).unit

    # def rotation_distance(self, q1, q2):
    #     delta_quat = self.to_quat(q2) * self.to_quat(q1).inverse
    #     return np.abs(delta_quat.angle)

    def update_reference_insim(self, curr_ref):
        if curr_ref.object is not None:
            # print(curr_ref.object)
            self.sim.model.site_pos[:] = curr_ref.object[:3]
            self.sim_obsd.model.site_pos[:] = curr_ref.object[:3]
            self.sim.forward()

    def get_obs_dict(self, sim):
        obs_dict = {}

        # get reference for current time (returns a named tuple)
        curr_ref = self.ref.get_reference(sim.data.time+self.motion_start_time)
        self.update_reference_insim(curr_ref)

        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['robot_err'] = obs_dict['qp'][:30] - curr_ref.robot
        obs_dict['object_pos_err'] = obs_dict['qp'][30:33] - curr_ref.object[:3]
        obs_dict['object_rot_err'] = obs_dict['qp'][33:] - quat2euler(curr_ref.object[3:])

        ## info about current hand pose + vel
        obs_dict['curr_hand_qpos'] = sim.data.qpos[:-6].copy() ## assuming only 1 object and the last values are posision + rotation
        obs_dict['curr_hand_qvel'] = sim.data.qvel[:-6].copy() ## not used for now

        ## info about target hand pose + vel
        obs_dict['targ_hand_qpos'] = obs_dict['curr_hand_qpos']  #temp for testing... TBD by Vikash get_reference()
        obs_dict['targ_hand_qvel'] = obs_dict['curr_hand_qvel']  #temp for testing... ## not used for now

        ## info about current object com + rotations
        # obs_dict['curr_obj_com'] = sim.named.data.xipos[self._object_name].copy() ## this should be with mujoco native
        # obs_dict['curr_obj_rot'] = sim.named.data.xquat[self._object_name].copy() ## this should be with mujoco native
        obs_dict['curr_obj_com'] = sim.data.get_body_xipos(self._object_name).copy()
        obs_dict['curr_obj_rot'] = sim.data.get_body_xquat(self._object_name).copy()

        ## info about target object com + rotations
        obs_dict['targ_obj_com']  = obs_dict['curr_obj_com']  #temp for testing...  # TBD by Vikash self.target_motion.object_pos
        obs_dict['targ_obj_rot']  = obs_dict['curr_obj_rot']  #temp for testing...  # TBD by Vikash self.target_motion.object_rot

        ## Errors
        obs_dict['hand_qpos_err'] = np.abs(obs_dict['curr_hand_qpos']-obs_dict['targ_hand_qpos'])
        obs_dict['hand_qvel_err'] = np.abs(obs_dict['curr_hand_qvel']-obs_dict['targ_hand_qvel'])

        obs_dict['obj_com_err'] =  np.abs(obs_dict['curr_obj_com'] - obs_dict['targ_obj_com'])
        # obs_dict['obj_rot_err'] =  self.rotation_distance(obs_dict['curr_obj_rot'], obs_dict['targ_obj_rot']) / np.pi

        return obs_dict


    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['robot_err'], axis=-1)
        far_th = 10

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',   pose_dist),
            ('bonus',   (pose_dist<1) + (pose_dist<2)),
            ('penalty', (pose_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<.5),
            ('done',    pose_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def playback(self):
        # import ipdb; ipdb.set_trace()
        idxs = self.ref.find_timeslot_in_reference(self.time)
        print(f"Time {self.time} {idxs} {self.ref.horizon}")
        ref_mot = self.ref.get_reference(self.time)
        rob_mot = ref_mot[1]
        obj_mot = ref_mot[2]
        self.sim.data.qpos[:len(rob_mot)] = rob_mot
        self.sim.data.qpos[len(rob_mot):len(rob_mot)+3] = obj_mot[:3]
        self.sim.forward()
        self.sim.data.time = self.sim.data.time + 0.02#self.env.env.dt

        return idxs[0] < self.ref.horizon-1

    def reset(self):
        self.ref.reset()
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
