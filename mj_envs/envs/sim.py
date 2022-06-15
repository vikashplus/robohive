import numpy as np
import mujoco
from mujoco import MjModel, MjData, mj_resetData, mj_step, GLContext, \
    mjr_render, mj_forward, MjvScene, MjrRect, MjvCamera, MjvOption, MjvPerturb, MjrContext
from typing import NamedTuple


class CameraMatrices(NamedTuple):
    """Component matrices used to construct the camera matrix.
    The matrix product over these components yields the camera matrix.
    Attributes:
      image: (3, 3) image matrix.
      focal: (3, 4) focal matrix.
      rotation: (4, 4) rotation matrix.
      translation: (4, 4) translation matrix.
    """
    image: np.ndarray
    focal: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


class Camera:
    def __init__(self,
                 physics,
                 height=240,
                 width=320,
                 camera_id=-1,
                 max_geom=None):
        buffer_width = physics.model.vis.global_.offwidth
        buffer_height = physics.model.vis.global_.offheight
        if width > buffer_width:
            raise ValueError(
                'Image width {} > framebuffer width {}. Either reduce '
                'the image width or specify a larger offscreen '
                'framebuffer in the model XML using the clause\n'
                '<visual>\n'
                '  <global offwidth="my_width"/>\n'
                '</visual>'.format(width, buffer_width))
        if height > buffer_height:
            raise ValueError(
                'Image height {} > framebuffer height {}. Either reduce '
                'the image height or specify a larger offscreen '
                'framebuffer in the model XML using the clause\n'
                '<visual>\n'
                '  <global offheight="my_height"/>\n'
                '</visual>'.format(height, buffer_height))
        if isinstance(camera_id, str):
            camera_id = mujoco.mj_name2id(physics.model, 7, camera_id)
            # camera_id = physics.model.name2id(camera_id, 'camera')
        if camera_id < -1:
            raise ValueError('camera_id cannot be smaller than -1.')
        if camera_id >= physics.model.ncam:
            raise ValueError(
                'model has {} fixed cameras. camera_id={} is invalid.'.
                    format(physics.model.ncam, camera_id))

        self._width = width
        self._height = height
        self._physics = physics

        # Variables corresponding to structs needed by Mujoco's rendering functions.
        # see https://github.com/deepmind/dm_control/blob/41d0c7383153f9ca6c12f8e865ef5e73a98759bd/dm_control/mujoco/wrapper/core.py#L668
        if max_geom is None:
            max_geom = _estimate_max_renderable_geoms(physics.model)
        self._scene = MjvScene(physics.model, max_geom)
        self._scene_option = MjvOption()

        self._perturb = MjvPerturb()
        self._perturb.active = 0
        self._perturb.select = 0

        self._rect = mujoco.MjrRect(0, 0, self._width, self._height)

        self._render_camera = MjvCamera()
        self._render_camera.fixedcamid = camera_id

        if camera_id == -1:
            self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        else:
            # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
            # camera explicitly defined in the model.
            self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Internal buffers.
        self._rgb_buffer = np.empty((self._height, self._width, 3),
                                    dtype=np.uint8)
        self._depth_buffer = np.empty((self._height, self._width),
                                      dtype=np.float32)

        self._gd_ctx = GLContext(width, height)

        # if self._physics.contexts.mujoco is not None:
        self._gd_ctx.make_current()
        self.mjr_context = MjrContext(
            self._physics.model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        mujoco.mjr_setBuffer(
                 mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value,
                 self.mjr_context)

    @property
    def width(self):
        """Returns the image width (number of pixels)."""
        return self._width

    @property
    def height(self):
        """Returns the image height (number of pixels)."""
        return self._height

    @property
    def option(self):
        """Returns the camera's visualization options."""
        return self._scene_option

    @property
    def scene(self):
        """Returns the `mujoco.MjvScene` instance used by the camera."""
        return self._scene

    def matrices(self):
        """Computes the component matrices used to compute the camera matrix.
        Returns:
          An instance of `CameraMatrices` containing the image, focal, rotation, and
          translation matrices of the camera.
        """
        camera_id = self._render_camera.fixedcamid
        if camera_id == -1:
            # If the camera is a 'free' camera, we get its position and orientation
            # from the scene data structure. It is a stereo camera, so we average over
            # the left and right channels. Note: we call `self.update()` in order to
            # ensure that the contents of `scene.camera` are correct.
            self.update()
            pos = np.mean([camera.pos for camera in self.scene.camera], axis=0)
            z = -np.mean([camera.forward for camera in self.scene.camera],
                         axis=0)
            y = np.mean([camera.up for camera in self.scene.camera], axis=0)
            rot = np.vstack((np.cross(y, z), y, z))
            fov = self._physics.model.vis.global_.fovy
        else:
            pos = self._physics.data.cam_xpos[camera_id]
            rot = self._physics.data.cam_xmat[camera_id].reshape(3, 3).T
            fov = self._physics.model.cam_fovy[camera_id]

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * self.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.width - 1) / 2.0
        image[1, 2] = (self.height - 1) / 2.0
        return CameraMatrices(
            image=image, focal=focal, rotation=rotation,
            translation=translation)

    @property
    def matrix(self):
        """Returns the 3x4 camera matrix.
        For a description of the camera matrix see, e.g.,
        https://en.wikipedia.org/wiki/Camera_matrix.
        For a usage example, see the associated test.
        """
        image, focal, rotation, translation = self.matrices()
        return image @ focal @ rotation @ translation

    def update(self, scene_option=None):
        """Updates geometry used for rendering.
        Args:
          scene_option: A custom `wrapper.MjvOption` instance to use to render
            the scene instead of the default.  If None, will use the default.
        """
        scene_option = scene_option or self._scene_option
        mujoco.mjv_updateScene(self._physics.model, self._physics.data,
                               scene_option, self._perturb,
                               self._render_camera,
                               mujoco.mjtCatBit.mjCAT_ALL,
                               self._scene)

    def _render_on_gl_thread(self, depth, overlays):
        """Performs only those rendering calls that require an OpenGL context."""

        # Render the scene.
        mujoco.mjr_render(self._rect, self._scene, self.mjr_context)
                          # self._physics.contexts.mujoco)

        if not depth:
            # If rendering RGB, draw any text overlays on top of the image.
            for overlay in overlays:
                raise RuntimeError
                overlay.draw(self._physics.contexts.mujoco, self._rect)

        # Read the contents of either the RGB or depth buffer.
        mujoco.mjr_readPixels(self._rgb_buffer if not depth else None,
                              self._depth_buffer if depth else None,
                              self._rect,
                              self.mjr_context)

    def render(
        self,
        overlays=(),
        depth=False,
        segmentation=False,
        scene_option=None,
        render_flag_overrides=None,
    ):

        if render_flag_overrides is None:
            render_flag_overrides = {}

        # Update scene geometry.
        self.update(scene_option=scene_option)

        # Enable flags to compute segmentation labels
        if segmentation:
            render_flag_overrides.update({
                mujoco.mjtRndFlag.mjRND_SEGMENT: True,
                mujoco.mjtRndFlag.mjRND_IDCOLOR: True,
            })

        # Render scene and text overlays, read contents of RGB or depth buffer.
        # with self.scene.override_flags(render_flag_overrides):
        self._gd_ctx.make_current()
        self._render_on_gl_thread(depth=depth,
                     overlays=overlays)

        if depth:
            # Get the distances to the near and far clipping planes.
            extent = self._physics.model.stat.extent
            near = self._physics.model.vis.map.znear * extent
            far = self._physics.model.vis.map.zfar * extent
            # Convert from [0 1] to depth in meters, see links below:
            # http://stackoverflow.com/a/6657284/1461210
            # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
            image = near / (1 - self._depth_buffer * (1 - near / far))
        elif segmentation:
            # Convert 3-channel uint8 to 1-channel uint32.
            image3 = self._rgb_buffer.astype(np.uint32)
            segimage = (image3[:, :, 0] +
                        image3[:, :, 1] * (2 ** 8) +
                        image3[:, :, 2] * (2 ** 16))
            # Remap segid to 2-channel (object ID, object type) pair.
            # Seg ID 0 is background -- will be remapped to (-1, -1).
            segid2output = np.full((self._scene.ngeom + 1, 2), fill_value=-1,
                                   dtype=np.int32)  # Seg id cannot be > ngeom + 1.
            visible_geoms = [g for g in self._scene.geoms if g.segid != -1]
            visible_segids = np.array([g.segid + 1 for g in visible_geoms],
                                      np.int32)
            visible_objid = np.array([g.objid for g in visible_geoms],
                                     np.int32)
            visible_objtype = np.array([g.objtype for g in visible_geoms],
                                       np.int32)
            segid2output[visible_segids, 0] = visible_objid
            segid2output[visible_segids, 1] = visible_objtype
            image = segid2output[segimage]
        else:
            image = self._rgb_buffer

        # The first row in the buffer is the bottom row of pixels in the image.
        return np.flipud(image)


# class Sim:
#     def __init__(self, model, data, frame_skip=1, width=244, height=244):
#         self.model = model
#         self.data = data
#         self.frame_skip = frame_skip
#         self.gl_context = None
#         self._width = width
#         self._height = height
#         self._rgb_buffer = np.empty((self._height, self._width, 3),
#                                     dtype=np.uint8)
#         self.camera = dict()
#
#     def step(self, action):
#         mj_step(self.model, self.data, nsteps=self.frame_skip)
#
#     def reset(self):
#         mj_resetData(self.model, self.data)
#
#     def forward(self):
#         mj_forward(self.model, self.data)
#
#     def render(self, width, height, mode, camera_name, device_id=0,
#                overlays=(), depth=False, scene_option=None,
#                render_flag_overrides=None, segmentation=False):
#         if not camera_name in self.camera:
#             self.camera[camera_name] = Camera(
#                 physics=self, height=height, width=width, camera_id=camera_name)
#         image = self.camera[camera_name].render(
#             overlays=overlays, depth=depth, segmentation=segmentation,
#             scene_option=scene_option,
#             render_flag_overrides=render_flag_overrides)
#         # no ops: https://github.com/deepmind/dm_control/blob/4e1a35595124742015ae0c7a829e099a5aa100f5/dm_control/mujoco/wrapper/core.py#L721
#         # camera._scene.free()  # pylint: disable=protected-access
#         return image
#
#     def __del__(self):
#         del self.gl_context


def get_model_and_data(model_path):
    model = MjModel.from_xml_path(model_path)
    data = MjData(model)
    return Sim(model, data)

def _estimate_max_renderable_geoms(model):
    """Estimates the maximum number of renderable geoms for a given model."""
    # Only one type of object frame can be rendered at once.
    max_nframes = max(
        [model.nbody, model.ngeom, model.nsite, model.ncam, model.nlight])
    # This is probably an underestimate, but it is unlikely that all possible
    # rendering options will be enabled simultaneously, or that all renderable
    # geoms will be present within the viewing frustum at the same time.
    return (
        3 * max_nframes +  # 1 geom per axis for each frame.
        4 * model.ngeom +  # geom itself + contacts + 2 * split contact forces.
        3 * model.nbody +  # COM + inertia box + perturbation force.
        model.nsite +
        model.ntendon +
        model.njnt +
        model.nu +
        model.nskin +
        model.ncam +
        model.nlight)

from dm_control.mujoco.engine import Physics as Sim
