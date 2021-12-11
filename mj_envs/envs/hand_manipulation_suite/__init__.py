from gym.envs.registration import register

print("RS:> Registering Hand Envs")
# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=100,
)
from mj_envs.envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=50,
)
from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

# Mix Tools environment
register(
    id='mixtools-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:MixToolsEnvV0',
    max_episode_steps=50,
)
from mj_envs.envs.hand_manipulation_suite.mixtools_v0 import MixToolsEnvV0

# Mix Tools environment
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
env_kwargs = {
            'model_paths' : [curr_dir + '/assets/TOOLS_ratchet.xml', curr_dir + '/assets/TOOLS_knife.xml',
                            curr_dir + '/assets/TOOLS_screwDriver.xml', curr_dir + '/assets/TOOLS_turner.xml',
                            curr_dir + '/assets/TOOLS_spatula.xml'],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]], # 0.05 0.0 0.4
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0, 0], [0, 0]], # TODO 3.1415927, 0, 0
            }
register(
    id='tools-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)
from mj_envs.envs.hand_manipulation_suite.tools_v0 import ToolsEnvV0

# env_kwargs = {
            # 'model_paths' : [curr_dir + '/assets/TOOLS_hammer.xml', curr_dir + '/assets/TOOLS_ratchet.xml', curr_dir + '/assets/TOOLS_knife.xml',
            # curr_dir + '/assets/TOOLS_screwDriver.xml', curr_dir + '/assets/TOOLS_turner.xml', curr_dir + '/assets/TOOLS_spatula.xml'],
            # }
env_kwargs = {
            'model_paths' : [curr_dir + '/assets/TOOLS_hammer.xml'],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]], # 0.05 0.0 0.4
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0, 0], [0, 0]], # TODO 3.1415927, 0, 0
            }
register(
    id='tools-test-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)

env_kwargs = {
            'model_paths' : [curr_dir + '/assets/TOOLS_ratchet.xml', curr_dir + '/assets/TOOLS_knife.xml',
                            curr_dir + '/assets/TOOLS_screwDriver.xml', curr_dir + '/assets/TOOLS_turner.xml',
                            curr_dir + '/assets/TOOLS_spatula.xml'],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]],
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0, 6.28], [0, 0]], # TODO 3.1415927, 0, 0
            #'tool_euler_range' : [[1.57, 1.57], [-1.57, 1.57], [0, 0]], # TODO 3.1415927, 0, 0
            }
register(
    id='tools-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)
env_kwargs = {
            'model_paths' : [curr_dir + '/assets/TOOLS_hammer.xml'],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]],
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0, 6.28], [0, 0]], # TODO 3.1415927, 0, 0
            #'tool_euler_range' : [[1.57, 1.57], [-1.57, 1.57], [0, 0]], # TODO 3.1415927, 0, 0
            }
register(
    id='tools-test-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)

env_kwargs = {
            'model_paths' : [curr_dir + '/assets/TOOLS_ratchet.xml', curr_dir + '/assets/TOOLS_knife.xml',
                            curr_dir + '/assets/TOOLS_screwDriver.xml', curr_dir + '/assets/TOOLS_turner.xml',
                            curr_dir + '/assets/TOOLS_spatula.xml'],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]],
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0.0, 0.4], [0, 0]], # TODO 3.1415927, 0, 0
            #'tool_euler_range' : [[1.57, 1.57], [-1.57, 1.57], [0, 0]], # TODO 3.1415927, 0, 0
            }
register(
    id='tools-demos-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)

env_kwargs = {
            'model_paths' : [
                             curr_dir + '/assets/TOOLS_hammer.xml',
                             curr_dir + '/assets/TOOLS_spoon.xml',
                             curr_dir + '/assets/TOOLS_fork.xml',
                             curr_dir + '/assets/TOOLS_flat_screwdriver.xml',
                            ],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]],
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0.0, 0.4], [0, 0]], # TODO 3.1415927, 0, 0
            #'tool_euler_range' : [[1.57, 1.57], [-1.57, 1.57], [0, 0]], # TODO 3.1415927, 0, 0
            }
register(
    id='tools-demos-test-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)

env_kwargs = {
            'model_paths' : [
                                curr_dir + '/assets/TOOLS_ratchet.xml',
                                curr_dir + '/assets/TOOLS_screwDriver.xml',
                                curr_dir + '/assets/TOOLS_turner.xml',
                                curr_dir + '/assets/TOOLS_spatula.xml',
                                curr_dir + '/assets/TOOLS_spoon.xml',
                                curr_dir + '/assets/TOOLS_flat_screwdriver.xml',
                            ],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]],
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0.0, 0.4], [0, 0]],
            #'tool_euler_range' : [[1.57, 1.57], [-1.57, 1.57], [0, 0]],
            }
register(
    id='tools-v2',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)
env_kwargs = {
            'model_paths' : [
                                curr_dir + '/assets/TOOLS_hammer.xml',
                                curr_dir + '/assets/TOOLS_fork.xml',
                                curr_dir + '/assets/TOOLS_knife.xml',
                            ],
            'target_pos_range' : [[0.05, 0.05], [0.0, 0.0], [0.1, 0.25]],
            'tool_pos_range' : [[-0.1, 0.1], [-0.2, -0.2], [0.035, 0.035]], # 0 -0.2 0.035
            'tool_euler_range' : [[1.57, 1.57], [0.0, 0.4], [0, 0]],
            #'tool_euler_range' : [[1.57, 1.57], [-1.57, 1.57], [0, 0]],
            }
register(
    id='tools-test-v2',
    entry_point='mj_envs.envs.hand_manipulation_suite:ToolsEnvV0',
    max_episode_steps=50,
    kwargs=env_kwargs,
)
