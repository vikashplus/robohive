""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Simulation using DeepMind Control Suite."""

import copy
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

import robohive.utils.import_utils as import_utils
from robohive.utils.prompt_utils import Prompt, prompt

import_utils.dm_control_isavailable()
import_utils.mujoco_isavailable()
import dm_control.mujoco as dm_mujoco

from robohive.physics.sim_scene import SimScene
from robohive.renderer.mj_renderer import MJRenderer


class DMSimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using dm_control."""

    def _normalize_xml_path(self, model_path: str) -> str:
        """Creates a MuJoCo-3-safe temporary main XML without changing sources.

        The compatibility issue here is limited to a subset of included XMLs.
        Keep the original asset tree in place and only rewrite the main XML plus
        any incompatible include files into a temporary location.
        """
        temp_dir = tempfile.mkdtemp(prefix='robohive_mjcf_')

        def normalize_top_level_defaults(tree: ET.ElementTree) -> bool:
            root = tree.getroot()
            if root.tag != 'mujocoinclude':
                return False

            changed = False
            for child in list(root):
                if child.tag == 'default' and child.get('class') is not None:
                    if child.get('class') == 'main':
                        del child.attrib['class']
                        changed = True
                        continue
                    wrapper = ET.Element('default')
                    root.insert(list(root).index(child), wrapper)
                    root.remove(child)
                    wrapper.append(child)
                    changed = True
            return changed

        def collect_default_classes(root: ET.Element) -> dict[str, ET.Element]:
            classes = {}
            for elem in root.iter('default'):
                default_class = elem.get('class')
                if default_class:
                    classes[default_class] = elem
            return classes

        def merge_default_nodes(base: ET.Element, extra: ET.Element):
            for child in list(extra):
                if child.tag == 'default' and child.get('class') is not None:
                    match = None
                    for base_child in list(base):
                        if base_child.tag == 'default' and base_child.get('class') == child.get('class'):
                            match = base_child
                            break
                    if match is None:
                        base.append(copy.deepcopy(child))
                    else:
                        merge_default_nodes(match, child)
                    continue

                match = None
                for base_child in list(base):
                    if base_child.tag == child.tag:
                        match = base_child
                        break

                if match is None:
                    base.append(copy.deepcopy(child))
                else:
                    match.attrib.update(child.attrib)

        def collapse_duplicate_defaults(root: ET.Element, parent_defaults: dict[str, ET.Element]) -> bool:
            changed = False

            def visit(parent: ET.Element):
                nonlocal changed
                for child in list(parent):
                    visit(child)
                    if child.tag != 'default':
                        continue

                    default_class = child.get('class')
                    if default_class is None or default_class not in parent_defaults:
                        continue

                    merge_default_nodes(parent_defaults[default_class], child)
                    parent.remove(child)
                    changed = True

            visit(root)
            return changed

        def absolutize_resource_paths(root: ET.Element, src_dir: str, fallback_dirs: tuple[str, ...] = ()) -> bool:
            changed = False
            candidate_dirs = []
            seen_dirs = set()

            def add_candidate_dir(path: str):
                path = os.path.abspath(path)
                if path not in seen_dirs:
                    candidate_dirs.append(path)
                    seen_dirs.add(path)

            for base_dir in (src_dir,) + fallback_dirs:
                base_dir = os.path.abspath(base_dir)
                current_dir = base_dir
                while True:
                    add_candidate_dir(current_dir)
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir == current_dir:
                        break
                    current_dir = parent_dir

            def resolve_existing_path(relative_path: str) -> str | None:
                for base_dir in candidate_dirs:
                    candidate = os.path.abspath(os.path.join(base_dir, relative_path))
                    if os.path.exists(candidate):
                        return candidate
                return None

            for elem in root.iter():
                if elem.tag == 'include' and 'file' in elem.attrib:
                    include_path = elem.get('file')
                    if include_path and not os.path.isabs(include_path):
                        resolved_include_path = resolve_existing_path(include_path)
                        if resolved_include_path is not None and resolved_include_path != include_path:
                            elem.set('file', resolved_include_path)
                            changed = True

                if elem.tag != 'include' and 'file' in elem.attrib:
                    file_path = elem.get('file')
                    if file_path and not os.path.isabs(file_path):
                        resolved_file_path = resolve_existing_path(file_path)
                        if resolved_file_path is not None and resolved_file_path != file_path:
                            elem.set('file', resolved_file_path)
                            changed = True

                if elem.tag == 'compiler':
                    for attr_name in ('assetdir', 'meshdir', 'texturedir'):
                        attr_value = elem.get(attr_name)
                        if attr_value and not os.path.isabs(attr_value):
                            resolved_dir = resolve_existing_path(attr_value)
                            if resolved_dir is not None and os.path.isdir(resolved_dir) and resolved_dir != attr_value:
                                elem.set(attr_name, resolved_dir)
                                changed = True

            return changed

        model_path = os.path.abspath(model_path)
        model_dir = os.path.dirname(model_path)
        model_tree = ET.parse(model_path)
        model_root = model_tree.getroot()
        known_defaults = collect_default_classes(model_root)
        absolutize_resource_paths(model_root, model_dir)

        include_index = 0
        for include in model_root.iter('include'):
            include_file = include.get('file')
            if not include_file:
                continue

            include_src = os.path.abspath(os.path.join(model_dir, include_file))
            include_tree = ET.parse(include_src)
            include_changed = normalize_top_level_defaults(include_tree)
            include_root = include_tree.getroot()
            include_changed = collapse_duplicate_defaults(include_root, known_defaults) or include_changed
            include_changed = absolutize_resource_paths(
                include_root,
                os.path.dirname(include_src),
                fallback_dirs=(os.path.dirname(os.path.dirname(include_src)), model_dir),
            ) or include_changed
            for default_class, default_elem in collect_default_classes(include_root).items():
                known_defaults.setdefault(default_class, default_elem)

            if include_changed:
                include_index += 1
                include_dst = os.path.join(temp_dir, f'include_{include_index}.xml')
                include_tree.write(include_dst, encoding='utf-8', xml_declaration=False)
                include.set('file', include_dst)
            else:
                include.set('file', include_src)

        normalized_root = os.path.join(temp_dir, os.path.basename(model_path))
        model_tree.write(normalized_root, encoding='utf-8', xml_declaration=False)
        return normalized_root

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: This can be a path to a Mujoco XML file, or an MJCF
                object.

        Returns:
            A dm_control Physics object.
        """
        if isinstance(model_handle, str):
            if model_handle.endswith('.xml'):
                sim = dm_mujoco.Physics.from_xml_path(self._normalize_xml_path(model_handle))
            elif isinstance(model_handle, str) and "<mujoco" in model_handle:
                sim = dm_mujoco.Physics.from_xml_string(model_handle)
            else:
                sim = dm_mujoco.Physics.from_binary_path(model_handle)
        else:
            raise NotImplementedError(model_handle)
        self._patch_mjmodel_accessors(sim.model)
        self._patch_mjdata_accessors(sim.data)
        return sim

    def advance(self, substeps: int = 1, render:bool = True):
        """Advances the simulation for one step."""
        # Step the simulation substeps (frame_skip) times.
        try:
            self.sim.step(substeps)
        except:
            prompt("Simulation couldn't be stepped as intended. Issuing a reset", type=Prompt.WARN)
            self.sim.reset()

        if render:
            # self.renderer.refresh_window()
            self.renderer.render_to_window()

    def _create_renderer(self, sim: Any) -> MJRenderer:
        """Creates a renderer for the given simulation."""
        return MJRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        # dm_control's MjModel defines __copy__.
        model_copy = copy.copy(self.model)
        self._patch_mjmodel_accessors(model_copy)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        self.model.save_binary(path)
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.contexts:
            logging.warning('No rendering context; not uploading height field.')
            return
        with self.sim.contexts.gl.make_current() as ctx:
            ctx.call(self.get_mjlib().mjr_uploadHField, self.model.ptr,
                     self.sim.contexts.mujoco.ptr, hfield_id)

    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        return dm_mujoco.wrapper.mjbindings.mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value.ptr

    def _patch_mjmodel_accessors(self, model):
        """Adds accessors to MjModel objects to support mujoco_py API.

        This adds `*_name2id` methods to a Physics object to have API
        consistency with mujoco_py.

        TODO(michaelahn): Deprecate this in favor of dm_control's named methods.
        """
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(model.ptr,
                                      mjlib.mju_str2Type(type_name.encode()),
                                      name.encode())
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(
                    type_name, name))
            return obj_id

        def get_xml():
            import os
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as td:
                filename = os.path.join(td, 'model.xml')
                ret = mjlib.mj_saveLastXML(filename.encode(), model.ptr)
                if ret == 0:
                    raise Exception('Failed to save XML')
                with open(filename, 'r') as f:
                    return f.read()

        if not hasattr(model, 'body_name2id'):
            model.body_name2id = lambda name: name2id('body', name)

        if not hasattr(model, 'geom_name2id'):
            model.geom_name2id = lambda name: name2id('geom', name)

        if not hasattr(model, 'site_name2id'):
            model.site_name2id = lambda name: name2id('site', name)

        if not hasattr(model, 'joint_name2id'):
            model.joint_name2id = lambda name: name2id('joint', name)

        if not hasattr(model, 'actuator_name2id'):
            model.actuator_name2id = lambda name: name2id('actuator', name)

        if not hasattr(model, 'camera_name2id'):
            model.camera_name2id = lambda name: name2id('camera', name)

        if not hasattr(model, 'sensor_name2id'):
            model.sensor_name2id = lambda name: name2id('sensor', name)

        if not hasattr(model, 'get_xml'):

            model.get_xml = lambda : get_xml()


    def _patch_mjdata_accessors(self, data):
        """Adds accessors to MjData objects to support mujoco_py API."""
        if not hasattr(data, 'body_xpos'):
            data.body_xpos = data.xpos

        if not hasattr(data, 'body_xquat'):
            data.body_xquat = data.xquat
