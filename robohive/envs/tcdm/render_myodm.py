import robohive
import gym
import time
import skvideo.io
import numpy as np
# from robohive import robohive_myodex_suite

for env_name in ['MyoHandGamecontrollerPass-v0']:#myosuite_myodex_suite:
    if 'Fixed' not in env_name and 'Random' not in env_name:
        env = gym.make(env_name)

        env.reset()
        frames=[]
        while env.playback():
            # env.mj_render()
            # time.sleep(.02)
            frames.append(env.sim.renderer.render_offscreen(
                                width=400,
                                height=400,
                                camera_id=2))
        env.close()
        print(f"Testing motion playback on: {env_name}  -- Frames # {np.asarray(frames).shape[0]}")
        # import ipdb; ipdb.set_trace()
        skvideo.io.vwrite('./videos/'+env_name+'.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
