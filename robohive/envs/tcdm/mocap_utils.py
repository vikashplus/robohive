#####################################################################################
# Copyright 2023 Vittorio Caggiano, Sudeep Dasari, Vikash Kumar
# Author	:: Vittorio Caggiano (caggiano@gmail.com),  Sudeep Dasari, Vikash Kumar
# License	:: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#####################################################################################

import numpy as np
import collections
from dm_env import specs
from dm_control.rl import control
from pyquaternion import Quaternion
from scipy.interpolate import interp1d


_DEFAULT_GLOBAL = Quaternion(axis=[0,0,1], angle=np.pi)
_MAX_TRIES = 10
_STEPS_PER_TRY=2000
_STOP_THRESH = 0.05
_OBJ_DELTA_THRESH = 0.001
_HUMAN_FPS = 120
_ADROIT_FPS=25


def _lerp(l, v1, v2):
    assert 0 <= l <= 1, 'l should be in [0,1]'
    return (1.0 - l) * v1 + l * v2


def _affine(X, old_origin, new_origin, R):
    X = X.copy() - old_origin
    X = np.matmul(R[None], X.reshape((-1, 3, 1)))[:,:,0]
    X += new_origin
    return X


def _obj_axis_to_quat(axis, global_rot=None):
    # norm of axis is angle
    angle = max(np.linalg.norm(axis), 1e-7)
    q = Quaternion(axis=-axis/angle, angle=angle).unit
    if global_rot is not None:
        q = global_rot * q
    return q


def _calc_start_pos(physics, object_name, rot_0, global_rot=_DEFAULT_GLOBAL):
    desired_q = _obj_axis_to_quat(rot_0, _DEFAULT_GLOBAL)
    n_tries = 0

    # reset physics to 0 and calculate deltas
    with physics.reset_context():
        physics.data.qpos[-6:] = 0
        physics.data.qvel[-6:] = 0
    physics.forward()
    # import ipdb; ipdb.set_trace()
    real_q = Quaternion(physics.named.data.xquat[object_name]).unit
    for _ in range(_STEPS_PER_TRY):
        physics.step()

    z = physics.data.qpos[-4]
    rz, ry, rx = (real_q.inverse * desired_q).yaw_pitch_roll

    # find initial stable pose
    while n_tries < _MAX_TRIES:
        with physics.reset_context():
            physics.data.qpos[-6:] = [0,0,z,rx,ry,rz]
            physics.data.qvel[-6:] = 0

        cur_pos = physics.data.qpos[-6:].copy()

        for _ in range(_STEPS_PER_TRY):
            physics.step()

        pos_err = np.linalg.norm(physics.data.qpos[-6:] - cur_pos)
        if  pos_err <= _STOP_THRESH and n_tries >= 1:
            break
        n_tries += 1
        z, rx, ry, rz = physics.data.qpos[-4:].copy()
        if n_tries < _MAX_TRIES / 4:
            rz, ry, rx = (real_q.inverse * desired_q).yaw_pitch_roll
    q0 = physics.data.qpos[-6:].copy()
    real_q = Quaternion(physics.named.data.xquat[object_name]).unit
    refined_COM = physics.named.data.xipos[object_name].copy()
    refined_global = (real_q * desired_q.inverse) * _DEFAULT_GLOBAL
    return q0, refined_COM, refined_global


def interpolate_traj(pos, ori, substeps):
    interp_pos, interp_quat = [], []
    t, n, step = 0, 0, 1.0 / substeps
    while n < substeps * (pos.shape[0] - 1):
        bot, top = int(np.floor(t)), int(np.ceil(t))
        interp_lambda = t - bot

        p = _lerp(interp_lambda, pos[bot], pos[top])
        q = Quaternion.slerp(to_quat(ori[bot]), to_quat(ori[top]),
                             interp_lambda)

        interp_pos.append(p[None])
        interp_quat.append(np.array([q.w, q.x, q.y, q.z])[None])
        t += step
        n += 1

    interp_pos = np.concatenate(interp_pos, 0)
    interp_quat = np.concatenate(interp_quat, 0)
    return interp_pos, interp_quat


class MoCapTask(control.Task):
    def __init__(self, mocap_controller, save_filename, length, input_name):
        self.save_filename = save_filename
        self._saver = []
        self._controller = mocap_controller
        self._n_steps = length
        self._input_name = input_name
        self._reference_motion ={
            'human_joint_coords': mocap_controller.human_joint_coords,
            'object_translation': mocap_controller.obj_trans,
            'object_orientation': mocap_controller.obj_orient,
            'human_obj_contact': mocap_controller.human_obj_contact,
            'grasp_frame': np.argmax(mocap_controller.human_obj_contact),
            'DATA_SUBSTEPS': 10
        }

    def action_spec(self, _):
        """Returns a `ArraySpec` representing mocap timestep"""
        return specs.BoundedArray((1,), np.float32, -10000, 10000)

    def initialize_episode(self, physics):
        """ Sets the state of the environment at the start of each episode.
            Called by `control.Environment` at the start of each episode *within*
            `physics.reset_context()` (see the documentation for `base.Physics`)

        Args:
            physics: An instance of `mujoco.Physics`.
        """
        self._last_t = 0
        with physics.reset_context():
            physics.data.qpos[:] = 0
            physics.data.qvel[:] = 0

        object_name = self._controller.object_name
        self.base_q = Quaternion(physics.named.data.xquat[object_name]).unit
        self.delta_xyz = physics.named.data.xipos[object_name] - physics.data.qpos[-6:-3]

        with physics.reset_context():
            physics.data.qpos[:] = 0
            physics.data.qpos[-6:] = self._controller.q0
            physics.data.qvel[:] = 0

        for _ in range(2000):
            self._controller.set_mocap(physics, 0)
            physics.step()

        obj_err = np.linalg.norm(physics.data.qpos[-6:] - self._controller.q0)
        if obj_err >= _STOP_THRESH:
            print("WARNING OBJ ERR EXCEEDS THRESH FOR", object_name)
            print('INPUT', self._input_name)
            print('TARGET', self._controller.q0)
            print("REAL", physics.data.qpos[-6:])
            print("ERR", obj_err)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        self._last_t = action
        self._controller.set_mocap(physics, action)

    def after_step(self, physics):
        """Called immediately after environment step: no-op by default"""
        objt = self._controller.obj_trans[self._last_t]
        objo = self._controller.obj_orient[self._last_t]
        desired_quat = Quaternion(w=objo[0], x=objo[1], y=objo[2], z=objo[3])
        x, y, z = objt - self.delta_xyz

        rz, ry, rx = (self.base_q.inverse * desired_quat).yaw_pitch_roll
        physics.data.qpos[-6:] = [x, y, z, rx, ry, rz]
        physics.forward()

    def get_observation(self, physics):
        """Returns dummy obs and appends state vector to saver"""
        # self._saver.append_from_physics(physics)
        self._saver.append(self.get_info_from_physics(physics))
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos.astype(np.float32).copy()
        obs['velocity'] = physics.data.qvel.astype(np.float32).copy()
        return obs

    def get_reward(self, _):
        # return a dummy reward
        return 0

    def get_frame(self):
        return physics.renderer.render_offscreen(camera_id=2)

    def get_termination(self, _):
        if len(self._saver) >= self._n_steps:
            # self._saver.save()
            # import ipdb; ipdb.set_trace()
            np.savez_compressed(self.save_filename, **self._reference_motion)
            return 0
        return None

    def get_info_from_physics(self, physics):
        qpos = physics.data.qpos.copy()
        qvel = physics.data.qvel.copy()
        # bodies = physics.body_poses
        # pos, rot, lv, av = bodies.pos, bodies.rot, bodies.linear_vel, bodies.angular_vel
        keys = ['s', 'sdot']#, 'eef_pos', 'eef_quat', 'eef_velp', 'eef_velr']
        data = [qpos, qvel]#, eef_pos, eef_quat, eef_linear_vel, eef_angular_vel]

        for key, arr in zip(keys, data):
            if self._reference_motion:
                prev_data = self._reference_motion.get(key, np.zeros([0] + list(arr.shape)))
                self._reference_motion[key] = np.concatenate((prev_data, arr[None]), 0)
            else:
                self._reference_motion[key] = arr[None]
        return [qpos, qvel]#, pos, rot, lv, av]


BODIES = ['wrist_point', 'ffknuckle_point', 'ffmiddle_point', 'ffdistal_point',
          'mfknuckle_point', 'mfmiddle_point', 'mfdistal_point', 'lfknuckle_point',
          'lfmiddle_point', 'lfdistal_point', 'rfknuckle_point', 'rfmiddle_point',
          'rfdistal_point', 'thknuckle_point', 'thmiddle_point', 'thdistal_point']
FINGERTIP_INDICES = [3, 6, 9, 12, 15]
BODY_TYPE = 'geom'

# MODELNAME = 'Full_Simplified_Wrist_Hand_v0.1.6'

def get_body_poses(physics):
    pos = []
    for bb in BODIES:
        pos.append(physics.named.data.geom_xpos[str(bb)][None])

    return np.array(pos)

class MoCapController(object):
    PARENTS = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

    def __init__(self, human_motion_file, physics, object_name, start_t, end_t, out_len):
        # initialize hand location data for human
        h_traj_raw = np.load(human_motion_file)
        h_traj = {'base_joints': h_traj_raw['base_joints']}

        old_x = np.array([i / float(_HUMAN_FPS) for i in range(end_t - start_t + 1)])
        new_x = np.clip(np.linspace(0, out_len / _ADROIT_FPS, num=out_len), 0, np.max(old_x))
        for k, v in h_traj_raw.items():
            if k == 'base_joints':
                continue
            f = interp1d(old_x, v[start_t:end_t+1], kind='cubic', axis=0)
            h_traj[k] = f(new_x)
        h_traj['rhand_contacts'] = h_traj['rhand_contacts'] >= 0.95
        self.human_obj_contact = h_traj['rhand_contacts']

        extra_resets = [0]
        obj = h_traj['obj_trans']
        obj_deltas = np.linalg.norm(obj[1:] - obj[:-1], axis=1) > _OBJ_DELTA_THRESH
        obj_starts = np.logical_and(np.logical_not(obj_deltas[:-1]), obj_deltas[1:])
        extra_resets += list(np.nonzero(obj_starts)[0])
        extra_resets = extra_resets + [np.argmax(self.human_obj_contact)]
        self.extra_resets = sorted(list(set(extra_resets)))

        self._human_joints = h_traj['base_joints'].reshape((16, 3))
        self._init_base_robot_pose(physics, self._human_joints.copy())

        q0, obj_xi0, global_quat = _calc_start_pos(physics, object_name, h_traj['obj_orient'][0])
        self._global_rot = global_quat.rotation_matrix.copy()
        self._q0, self._obj_name = q0, object_name

        # grab pose and translational data from human motion file
        self._rh_pose = h_traj['rh_pose']
        self._rh_trans = _affine(h_traj['rh_trans'], h_traj['obj_trans'][0],
                                 obj_xi0, self._global_rot)

        # transform human joints into simulated coordinate space
        joints_flat = h_traj['rh_joints'].reshape((out_len * 16, 3))
        self.human_joint_coords = _affine(joints_flat, h_traj['obj_trans'][0],
                                  obj_xi0, self._global_rot).reshape((out_len, 16, 3))

        # transform object data into simulated coordinate space
        self.obj_trans = _affine(h_traj['obj_trans'], h_traj['obj_trans'][0],
                                 obj_xi0, self._global_rot)
        obj_orient = []
        for ori in h_traj['obj_orient']:
            rot = _obj_axis_to_quat(ori, global_rot=global_quat)
            rot = np.array([rot.w, rot.x, rot.y, rot.z])
            obj_orient.append(rot[None])
        self.obj_orient = np.concatenate(obj_orient, 0)

    def _init_base_robot_pose(self, physics, hand_joints):
        # calculate hand pose link directions
        link_directions = hand_joints[1:] - hand_joints[self.PARENTS[1:]]
        link_directions /= np.linalg.norm(link_directions, axis=1, keepdims=True)

        # estimate adroit hand lenths
        # b_j = physics.body_poses.pos
        b_j = get_body_poses(physics).squeeze()
        base_deltas = np.linalg.norm(b_j[1:] - b_j[self.PARENTS[1:]],
                                     axis=1, keepdims=True)
        base_deltas = link_directions * base_deltas

        for i in range(1, len(self.PARENTS)):
            b_j[i] = b_j[self.PARENTS[i]] + base_deltas[i-1]
        self._robot_joints = b_j

    def _fk(self, base_translation, joint_angles):
        # initialize array of deltas between joints and parent joint
        deltas = self._robot_joints.copy()
        deltas[1:] -= deltas[self.PARENTS[1:]]

        # set robot root joint to global_rot * root_human_joint
        human_base_joint = self._human_joints[0].reshape((3, 1))
        deltas[0] = self._global_rot.dot(human_base_joint)[:,0]

        # calculate rotation matrices that match joint angles
        rots = [axis_angle_to_rot(r) for r in joint_angles]
        rots = [self._global_rot.dot(rots[0])] + [r for r in rots[1:]]

        # create transform matrix and perform forward kinematics
        transforms = [to_transform_mat(r, t) for r, t in zip(rots, deltas)]
        for i in range(1, len(transforms)):
            transforms[i] = transforms[self.PARENTS[i]].dot(transforms[i])

        # copy out final joint angle locations and transform w/ base_translation
        final_joints = np.concatenate([t[:3, 3].reshape((1, 3)) for t in transforms], 0)
        return final_joints + base_translation.reshape((1, 3))

    def set_mocap(self, physics, time):
        # intialize time variables and slerp lambda
        assert 0 <= time < self.T, "time should be in [0, T)"
        bot, top = int(np.floor(time)), int(np.ceil(time))
        interp_lambda = time - bot

        # calculate values using _lerp
        base_trans = _lerp(interp_lambda, self._rh_trans[bot], self._rh_trans[top])
        hand_pose = _lerp(interp_lambda, self._rh_pose[bot], self._rh_pose[top])
        target_pos = self._fk(base_trans, hand_pose)

        # put target pos into mocap tensors
        for i, p in enumerate(target_pos):
            physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p

    @property
    def T(self):
        return self._rh_pose.shape[0]

    @property
    def q0(self):
        return self._q0.copy()

    @property
    def object_name(self):
        return self._obj_name




import numpy as np
from pyquaternion import Quaternion


def to_quat(arr):
    if isinstance(arr, Quaternion):
        return arr.unit
    if len(arr.shape) == 2:
        return Quaternion(matrix=arr).unit
    elif len(arr.shape) == 1 and arr.shape[0] == 9:
        return Quaternion(matrix=arr.reshape((3,3))).unit
    return Quaternion(array=arr).unit


def rotation_distance(q1, q2):
    delta_quat = to_quat(q2) * to_quat(q1).inverse
    return np.abs(delta_quat.angle)


def root_to_point(root_pos, root_rotation, point):
    if isinstance(root_rotation, Quaternion) \
                  or root_rotation.shape != (3,3):
        root_rotation = to_quat(root_rotation).rotation_matrix
    root_rotation_inv = root_rotation.T
    delta = (point - root_pos).reshape((3,1))
    return root_rotation_inv.dot(delta).reshape(-1)


def to_transform_mat(R, t):
    pad = np.array([0, 0, 0, 1]).astype(np.float32).reshape((1, 4))
    Rt = np.concatenate((R, t.reshape((3, 1))), 1)
    return np.concatenate((Rt, pad), 0)


def axis_angle_to_rot(axis_angle):
    angle = max(1e-8, np.linalg.norm(axis_angle))
    axis = axis_angle / angle
    quat = Quaternion(axis=axis, angle=angle)
    return quat.rotation_matrix


# class Pose(object):
#     def __init__(self, pos, rotation):
#         assert len(pos.shape) == 2, "pos should be batched"
#         assert len(rotation.shape) >= 2, "rotation should be batched"
#         assert pos.shape[0] == rotation.shape[0], "Batch sizes should match"
#         self.pos = pos
#         self.rot = rotation

#     def __len__(self):
#         return self.pos.shape[0]


# class PoseAndVelocity(Pose):
#     def __init__(self, pos, rotation, linear_vel, angular_vel):
#         super().__init__(pos, rotation)
#         assert len(linear_vel.shape) == 2, "linear_vel should be batched"
#         assert len(angular_vel.shape) == 2, "angular_vel should be batched"
#         assert pos.shape[0] == angular_vel.shape[0], "Batch sizes should match"
#         assert linear_vel.shape[0] == angular_vel.shape[0], "Batch sizes should match"
#         self.linear_vel = linear_vel
#         self.angular_vel = angular_vel
