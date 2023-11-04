import robohive
import gym
import numpy as np
from mocap_utils import MoCapController, MoCapTask
import dm_env
from mocap_utils import BODIES
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('save_file', type=str, help='filename to save')
parser.add_argument('--sim_name', type=str, default='table_apple')
parser.add_argument('--human_motion_file', type=str, default='data/human_trajs/apple_eat0.npz')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=999)
parser.add_argument('--length', type=int, default=100)
parser.add_argument('--substeps', type=int, default=10)
parser.add_argument('--no_viewer', action='store_true')

# python playback_mocap.py --human_motion_file /Users/caggiano/Dropbox (Meta)/mimic/mimic_dataset_mocap/data/human_trajs/s1/airplane_pass_1.npz ./data_myoHand/ik/s1/airplane_lift_myo.npz --start 163 --end 484 --sim_name MyoHandAirplanePass-v0 --length 68 --no_viewer

if __name__ == '__main__':

    args = parser.parse_args()
    args.end = args.end if args.end is not None else args.start + args.length

    # envMyoSuit=gym.make('MyoHandAirplanePass-v0')
    envMyoSuit=gym.make(args.sim_name)
    physics_myo=envMyoSuit.env.sim
    # jx_mocap coordinates
    # for i,b in enumerate(BODIES): #print(b,i, physics_myo.sim.named.data.xipos[b])
    #     print(f"<body mocap=\"true\" name=\"j{i}_mocap\" pos=\"{' '.join(map(str, physics_myo.sim.named.data.xipos[b]))} \" ><site name=\"j{i}\" size=\"0.015\" rgba=\"0 0 1 0.5\" pos=\"0 0 0\"/></body>\" ")

    # human_motion_file = '/Users/caggiano/Dropbox (Meta)/mimic/mimic_dataset_mocap/data/human_trajs/'+'s1/airplane_pass_1.npz'
    human_motion_file = args.human_motion_file
    f_name = human_motion_file.split('/')[-1]
    object_name, task_name = f_name[:-4].split('_')[:2]

    traj = np.load(human_motion_file)

    start, end, length = args.start, args.end, args.length
    # start = 163
    # end   = 484
    # length= 68

    mocap = MoCapController(human_motion_file, physics_myo.sim, object_name, start, end, length)

    task = MoCapTask(mocap, args.save_file, length, human_motion_file)


    def mocap_policy(step):
        global t, time_step
        if step.first():
            t = 0
        t += 1
        return t

    from dm_control.rl import control
    env = control.Environment(physics_myo.sim, task, n_sub_steps=10) # controller accounts for extra substeps


    def _render(physics, save_file):
        import imageio
        data = np.load(save_file)
        robot_file = save_file[:-4] + '_render_myo.mp4'

        for i in range(16):
            physics.named.data.mocap_pos['j{}_mocap'.format(i)] = -1
        out = imageio.get_writer(robot_file, fps=25)
        for s in data['s']:
            physics.data.qpos[:] = s
            physics.forward()
            im = physics.render(height=240 * 2, width=320 * 2, camera_id=2)
            out.append_data(im)
        out.close()

    v = []
    if args.no_viewer:
        step = env.reset()
        count=0
        while not step.last():
            ac = mocap_policy(step)
            # try:
            step = env.step(ac)
            # print(count)
            count +=1
            v.append(physics_myo.sim.named.data.qpos[['ARTx','ARTy','ARTz']])
            # except:
            #     print('error for input', args.human_motion_file)
            #     exit(1)
        _render(physics_myo.sim, args.save_file)
    else:
        from dm_control import viewer
        viewer.launch(env, policy=mocap_policy)
