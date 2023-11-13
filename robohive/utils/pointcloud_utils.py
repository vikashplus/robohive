import numpy as np
from robohive.utils.quat_math import mulQuat, euler2quat, quat2mat


# ------ MuJoCo specific functions ------

def get_point_cloud(env, depth, camera_name):
    # Make sure the depth values are in meters. If the depth comes 
    # from robohive, it is already in meters. If it directly comes 
    # from mujoco, you need to use the convert_depth function below. 
    # Output is flattened. Each row is a point in 3D space.
    fovy = env.sim.model.cam_fovy[env.sim.model.camera_name2id(camera_name)]
    K = get_intrinsics(fovy, depth.shape[0], depth.shape[1])
    pc = depth2xyz(depth, K)
    pc = pc.reshape(-1, 3)

    transform = get_extrinsics(env, camera_name=camera_name)
    new_pc = np.ones((pc.shape[0], 4))
    new_pc[:, :3] = pc
    new_pc = (transform @ new_pc.transpose()).transpose()
    return new_pc[:, :-1]


def convert_depth(env, depth):
    # Convert raw depth values into meters
    # Check this as well: https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L734
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    depth_m = depth * 2 - 1
    depth_m = (2 * near * far) / (far + near - depth_m * (far - near))
    return depth_m


def get_extrinsics(env, camera_name):
    # Transformation from camera frame to world frame
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.model.cam_pos[cam_id]
    cam_quat = env.sim.model.cam_quat[cam_id]
    cam_quat = mulQuat(cam_quat, euler2quat([np.pi, 0, 0]))
    return get_transformation_matrix(cam_pos, cam_quat)


def get_transformation_matrix(pos, quat):
    # Convert the pose from MuJoCo format to a 4x4 transformation matrix
    arr = np.identity(4)
    arr[:3, :3] = quat2mat(quat)
    arr[:3, 3] = pos
    return arr


# ------ General functions ------

def get_intrinsics(fovy, img_width, img_height):
    # Get the camera intrinsics matrix
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
    # Convert depth image to point cloud
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
    # Visualize a point cloud using open3d
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
        # Visualize coordinate frame
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([mesh, pcd])
    else:
        o3d.visualization.draw_geometries([pcd])

