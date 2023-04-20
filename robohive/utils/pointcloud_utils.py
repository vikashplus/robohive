import numpy as np
from utils.rotations import *
import cv2


# -------------------- Generic ----------------------------
def get_intrinsics(fovy, img_width, img_height):
    # fovy = self.sim.model.cam_fovy[cam_no]
    aspect = float(img_width) / img_height
    fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy) * 0.5) * aspect)
    fovx = np.rad2deg(fovx)
    cx = img_width / 2.
    cy = img_height / 2.
    fx = cx / np.tan(np.deg2rad(fovx / 2.))
    fy = cy / np.tan(np.deg2rad(fovy / 2.))
    K = np.zeros((3,3), dtype=np.float64)
    K[2][2] = 1
    K[0][0] = fx
    K[1][1] = fy
    K[0][2] = cx
    K[1][2] = cy
    return K


def depth2xyz(depth, cam_K):
    h, w = depth.shape
    ymap, xmap = np.meshgrid(np.arange(w), np.arange(h))

    x = ymap
    y = xmap
    z = depth

    x = (x - cam_K[0,2]) * z / cam_K[0,0]
    y = (y - cam_K[1,2]) * z / cam_K[1,1]

    xyz = np.stack([x, y, z], axis=2)
    return xyz


def visualize_point_cloud_from_nparray(d, c=None, vis_coordinate=False):
    if c is not None:
        if len(c.shape) == 3:
            c = c.reshape(-1, 3)
        if c.max() > 1:
            c = c.astype(np.float64)/256

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d)
    if c is not None:
        pcd.colors = o3d.utility.Vector3dVector(c)

    if vis_coordinate:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([mesh, pcd])
    else:
        o3d.visualization.draw_geometries([pcd])


# -------------------- MuJoCo Specific ----------------------------
def get_transformation_matrix(pos, quat):
    arr = np.identity(4)
    arr[:3, :3] = quat2mat(quat)
    arr[:3, 3] = pos
    return arr


def get_transformation(env, camera_name=None):
    if camera_name is None:
        camera_name = env.camera_names[0]
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.model.cam_pos[cam_id]
    cam_quat = env.sim.model.cam_quat[cam_id]
    cam_quat = quat_mul(cam_quat, euler2quat([np.pi, 0, 0]))
    return get_transformation_matrix(cam_pos, cam_quat)


def convert_depth(env, depth):
    # Convert depth into meter
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    depth_m = depth * 2 - 1
    depth_m = (2 * near * far) / (far + near - depth_m * (far - near))
    # Check this as well: https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L734
    return depth_m


def get_object_point_cloud(env, depth, img):
    depth = convert_depth(env, depth)
    full_pc = get_point_cloud(env, depth)
    obj_mask = get_obj_mask(img)
    pc = full_pc[obj_mask.reshape(-1),:]
    return pc


def get_point_cloud(env, depth, camera_name=None):
    # make sure to convert the raw depth image from MuJoCo using convert_depth
    # output is flattened
    if camera_name is None:
        camera_name = env.camera_names[0]
    fovy = env.sim.model.cam_fovy[env.sim.model.camera_name2id(camera_name)]
    K = get_intrinsics(fovy, depth.shape[0], depth.shape[1])
    pc = depth2xyz(depth, K)
    pc = pc.reshape(-1, 3)

    transform = get_transformation(env, camera_name=camera_name)
    new_pc = np.ones((pc.shape[0], 4))
    new_pc[:, :3] = pc
    new_pc = (transform @ new_pc.transpose()).transpose()
    return new_pc[:, :-1]

