""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Rendering simulation using dm_control."""

import sys

import numpy as np
import dm_control.mujoco as dm_mujoco
import dm_control.viewer as dm_viewer
import dm_control._render as dm_render

from typing import Union

from mj_envs.renderer.renderer import Renderer, RenderMode

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 640
DEFAULT_WINDOW_HEIGHT = 480

# Default window title.
DEFAULT_WINDOW_TITLE = 'MuJoCo Viewer'

# Internal renderbuffer size, in pixels.
_MAX_RENDERBUFFER_SIZE = 2048


class DMRenderer(Renderer):
    """Renders DM Control Physics objects."""

    def __init__(self, sim: dm_mujoco.Physics):
        super().__init__(sim)
        self._window = None

    def render_to_window(self):
        """Renders the Physics object to a window.

        The window continuously renders the Physics in a separate thread.

        This function is a no-op if the window was already created.
        """
        if not self._window:
            self._window = DMRenderWindow()
            self._window.load_model(self._sim)
            self._update_camera_properties(self._window.camera)

        self._window.run_frame()

    def refresh_window(self):
        """Refreshes the rendered window if one is present."""
        if self._window is None:
            return
        self._window.run_frame()

    def render_offscreen(self,
                         width: int,
                         height: int,
                        #  mode: RenderMode = RenderMode.RGB,
                         rgb: bool = True,
                         depth: bool = False,
                         segmentation: bool = False,
                         camera_id: Union[int, str] = -1,
                         device_id=-1) -> np.ndarray:
        """Renders the camera view as a numpy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A numpy array of the pixels.
        """
        assert width > 0 and height > 0

        if isinstance(camera_id, str):
            camera_id = self._sim.model.name2id(camera_id, 'camera')
        elif camera_id == None:
            camera_id = -1

        # TODO(michaelahn): Consider caching the camera.
        camera = dm_mujoco.Camera(
            physics=self._sim, height=height, width=width, camera_id=camera_id)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera_properties(camera._render_camera)  # pylint: disable=protected-access

        # image = camera.render(
        #    depth=(mode == RenderMode.DEPTH),
        #    segmentation=(mode == RenderMode.SEGMENTATION))
        rgb_arr = None; dpt_arr = None; seg_arr = None
        if rgb:
            rgb_arr = camera.render()
            # rgb_arr = rgb_arr[::-1, :, :]
        if depth:
            dpt_arr = camera.render(depth=True)
            dpt_arr = dpt_arr[::-1, :]
        if segmentation:
            seg_arr = camera.render(segmentation=True)
            seg_arr = seg_arr[::-1, :]

        camera._scene.free()  # pylint: disable=protected-access
        if depth and segmentation:
            return rgb_arr, dpt_arr, seg_arr
        elif depth:
            return rgb_arr, dpt_arr
        elif segmentation:
            return rgb_arr, seg_arr
        else:
            return rgb_arr


    def close(self):
        """Cleans up any resources being used by the renderer."""
        if self._window:
            self._window.close()
            self._window = None


class DMRenderWindow:
    """Class that encapsulates a graphical window."""

    def __init__(self,
                 width: int = DEFAULT_WINDOW_WIDTH,
                 height: int = DEFAULT_WINDOW_HEIGHT,
                 title: str = DEFAULT_WINDOW_TITLE):
        """Creates a graphical render window.

        Args:
            width: The width of the window.
            height: The height of the window.
            title: The title of the window.
        """
        self._viewport = dm_viewer.renderer.Viewport(width, height)
        self._window = dm_viewer.gui.RenderWindow(width, height, title)
        self._viewer = dm_viewer.viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard)
        self._draw_surface = None
        self._renderer = dm_viewer.renderer.NullRenderer()

    @property
    def camera(self):
        return self._viewer._camera._camera  # pylint: disable=protected-access

    def close(self):
        self._viewer.deinitialize()
        self._renderer.release()
        self._draw_surface.free()
        self._window.close()

    def load_model(self, physics):
        """Loads the given Physics object to render."""
        self._viewer.deinitialize()

        self._draw_surface = dm_render.Renderer(
            max_width=_MAX_RENDERBUFFER_SIZE, max_height=_MAX_RENDERBUFFER_SIZE)
        self._renderer = dm_viewer.renderer.OffScreenRenderer(
            physics.model, self._draw_surface)

        self._viewer.initialize(physics, self._renderer, touchpad=False)

    def run_frame(self):
        """Renders one frame of the simulation.

        NOTE: This is extremely slow at the moment.
        """
        # pylint: disable=protected-access
        glfw = dm_viewer.gui.glfw_gui.glfw
        glfw_window = self._window._context.window
        if glfw.window_should_close(glfw_window):
            sys.exit(0)

        self._viewport.set_size(*self._window.shape)
        self._viewer.render()
        pixels = self._renderer.pixels

        with self._window._context.make_current() as ctx:
            ctx.call(self._window._update_gui_on_render_thread, glfw_window,
                     pixels)
        self._window._mouse.process_events()
        self._window._keyboard.process_events()
        # pylint: enable=protected-access
