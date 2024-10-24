""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """


DESC = '''
Helper script to render images offscreen and save using a mujoco model.\n
USAGE:\n
    $ python render_cams.py --model_path <../model.xml> --cam_names <name1> --cam_names <name2> \n
EXAMPLE:\n
    $ python robohive/tutorials/render_cams.py -m robohive/envs/arms/franka/assets/franka_reach_v0.xml -c left_cam  -c top_cam
    $ python robohive/tutorials/render_cams.py -m robohive/simhive/robel_sim/dkitty/kitty-v2.1.xml -c "A:trackingY" -c "A:trackingZ"

'''

import mujoco

from PIL import Image
import click

@click.command(help=DESC)
@click.option('-m', '--model_path', required=True, type=str, help='model file')
@click.option('-c', '--cam_names', required=True, multiple=True, help=('Camera names for rendering'))
@click.option('-w', '--width', type=int, default=640, help='image width')
@click.option('-h', '--height', type=int, default=480, help='image height')
@click.option('-d', '--device_id', type=int, default=0, help='device id for rendering')

def main(model_path, cam_names, width, height, device_id):

    # prepare model, data, scene
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # prepare the renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # save images
    for i, cam in enumerate(cam_names):
        # update the scene
        renderer.update_scene(mj_data, camera=cam)
        # render the rgb_array
        rgb_arr = renderer.render()
        # save the image
        image = Image.fromarray(rgb_arr)
        image.save(cam+".jpeg")
        print("saved "+cam+".jpeg")

if __name__ == '__main__':
    main()