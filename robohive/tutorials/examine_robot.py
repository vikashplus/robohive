""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

DESC = '''
Tutorial: Demonstrate how to use the RoboHive's Robot class in isolation. This usecase is common in scenarios where the entire env definitions aren't required for experiments. In this tutorial we demonstate how we can use the RoboHive's Franka Robot in the real world using specifications available in a hardware config file.
USAGE Help:\n
    $ python tutorials/examine_robot.py --sim_path PATH_robot_sim.xml --config_path PATH_robot_configurations.config\n
EXAMPLE:\n
    $ python tutorials/examine_robot.py --sim_path envs/arms/franka/assets/franka_reach_v0.xml --config_path envs/arms/franka/assets/franka_reach_v0.config
'''

from robohive.robot.robot import Robot
import numpy as np
import click


@click.command(help=DESC)
@click.option('-sp', '--sim_path', type=str, help='environment to load', required= True, default='envs/arms/franka/assets/franka_reach_v0.xml')
@click.option('-cp', '--config_path', type=str, help='Config to load', required= True, default='envs/arms/franka/assets/franka_reach_v0.config')
@click.option('-ih', '--is_hardware', type=bool, help='Use on real robot hardware', default=False)
@click.option('-ja', '--jnt_amp', type=float, help='Range for random poses. 0:mean-pose. 1:full joint range', default=.15)
@click.option('-fs', '--frame_skip', type=int, help='hardware_dt = frame_skip*sim_dt', default=40)
@click.option('-th', '--traj_horizon', type=int, help='Trajectory duration in seconds', default=4)
@click.option('-lr', '--live_render', type=bool, help='Open a rendering window?', default=True)
def main(sim_path, config_path, is_hardware, jnt_amp, frame_skip, traj_horizon, live_render):

    # start robots and visualizers
    robot = Robot(robot_name="Robot Demo", model_path=sim_path, config_path=config_path, act_mode='pos', is_hardware=is_hardware)
    sim = robot.sim
    render_cbk = sim.renderer.render_to_window if live_render else None

    # derived variables
    traj_dt = frame_skip*sim.model.opt.timestep
    traj_nsteps = int(traj_horizon/traj_dt)
    jnt_mean = np.mean(sim.model.jnt_range, axis=1)
    djnt_mean = np.zeros(sim.model.nv)
    jnt_rng = 0.5*(sim.model.jnt_range[:,1]-sim.model.jnt_range[:,0])

    # Goto joint targets
    for iloop in range(10):
        print(f"Loop ID:{iloop}")

        # Sample a new desired position
        des_jnt_pos = jnt_mean + jnt_amp*np.random.uniform(high=jnt_rng, low=-jnt_rng)
        act = des_jnt_pos

        # execute on robot
        robot.reset(reset_pos=jnt_mean, reset_vel=djnt_mean)
        sensors = robot.get_sensors() # gets latest sensors and propage it in the sim
        for _ in range(traj_nsteps):
            robot.step(ctrl_desired=act, step_duration=traj_dt, ctrl_normalized=False, realTimeSim=(live_render==True), render_cbk=render_cbk)
            sensors = robot.get_sensors() # gets current sensors and propage it in the sim

        # Report progress
        jnt_err_0 = des_jnt_pos-jnt_mean
        jnt_err_t = des_jnt_pos-sim.data.qpos
        print("Mean joint pose error:: t[0]={:.3f} => t[{}]={:.3f} rad".format(np.linalg.norm(jnt_err_0), traj_nsteps, np.linalg.norm(jnt_err_t)))

if __name__ == '__main__':
    main()
