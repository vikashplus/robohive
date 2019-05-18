import numpy as np
import time as timer
import pickle


def init(gym_env, use_tactile, camera_name=None, frame_size=(128, 128)):
    gym_env.reset()
    robot_info = gym_env.env.env.get_proprioception(use_tactile=use_tactile)

    o = gym_env.env.env.get_pixels(camera_name=camera_name, frame_size=frame_size)

    d = False
    t = 0
    prev_o = o
    prev_prev_o = o

    return o, prev_o, prev_prev_o, t, d, robot_info


def single_step(gym_env, policy, o, prev_o, prev_prev_o, robot_info, mode,
                camera_name,
                use_tactile, frame_size=(128, 128)):
    o_c = o
    prev_o_c = prev_o
    prev_prev_o = np.expand_dims(prev_prev_o, axis=0)
    prev_o = np.expand_dims(prev_o, axis=0)
    o = np.expand_dims(o, axis=0)

    o = np.concatenate((prev_prev_o, prev_o, o), axis=0)
    a = policy.get_action(o, robot_info=robot_info)[0] if mode == 'exploration' else \
        policy.get_action(o, robot_info=robot_info)[1]['evaluation']
    # Maybe clip here too?

    prev_prev_o = prev_o_c
    prev_o = o_c
    _, r, d, env_info = gym_env.step(a)

    robot_info = gym_env.env.env.get_proprioception(use_tactile=use_tactile)

    o = gym_env.env.env.get_pixels(camera_name=camera_name, frame_size=frame_size)

    return o, r, d, env_info, prev_o, prev_prev_o, robot_info


def visualize_policy_offscreen(gym_env, policy, use_tactile, horizon=1000,
                               num_episodes=1,
                               frame_size=(640, 480),
                               frame_size_model=(128, 128),
                               mode='exploration',
                               save_loc='/tmp/',
                               filename='newvid',
                               camera_name=None,
                               pickle_dump=True):
    import skvideo.io
    for ep in range(num_episodes):
        print("Episode %d: rendering offline " % ep, end='', flush=True)
        arrs = []
        t0 = timer.time()
        o, prev_o, prev_prev_o, t, d, robot_info = init(gym_env, use_tactile, camera_name,
                                                        frame_size_model)
        while t < horizon and d is False:
            o, r, d, env_info, prev_o, prev_prev_o, robot_info = single_step(gym_env, policy,
                                                                             o, prev_o, prev_prev_o,
                                                                             robot_info,
                                                                             mode, camera_name, use_tactile,
                                                                             frame_size_model)
            t = t + 1
            curr_frame = gym_env.env.env.sim.render(width=frame_size[0], height=frame_size[1],
                                                    mode='offscreen', camera_name=camera_name, device_id=0)
            arrs.append(curr_frame[::-1, :, :])
            print(t, end=', ', flush=True)
        file_name = save_loc + filename + str(ep) + ".mp4"
        if not pickle_dump:
            skvideo.io.vwrite(file_name, np.asarray(arrs))
        else:
            file_name += '.pickle'
            pickle.dump(np.asarray(arrs), open(file_name, 'wb'))
        print("saved", file_name)
        t1 = timer.time()
        print("time taken = %f" % (t1 - t0))
