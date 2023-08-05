import robohive
import gym
import time
import click

DESC="""
Script to render trajectories embeded in the env"
"""

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default="AdroitBananaPass-v0")
@click.option('-h', '--horizon', type=int, help='playback horizon', default=-1)
@click.option('-n', '--num_playback', type=int, help='Number of time to loop playback', default=1)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'none']), help='visualize onscreen?', default='onscreen')
def playback_default_traj(env_name, horizon, num_playback, render):
    env = gym.make(env_name)
    if horizon==-1: horizon = env.horizon

    # Start playback loops
    for n in range(num_playback):
        print(f"Playback loop: {n}")
        env.reset()

        # Rollout a traj
        for h in range(horizon):
            env.playback()

            # render onscreen if asked
            if render=='onscreen':
                env.mj_render()
                time.sleep(.1)
    env.close()


if __name__ == '__main__':
    playback_default_traj()
