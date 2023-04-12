def mujoco_py_isavailable():
    try:
        import mujoco_py
    except ImportError as e:
        help = """
        Options:
            (1) follow setup instructions here: https://github.com/openai/mujoco-py/
            (2) install mujoco_py via pip (pip install mujoco_py)
            (3) install free_mujoco_py via pip (pip install free-mujoco-py)
        """
        raise ModuleNotFoundError(f"{e}. {help}")

def mujoco_isavailable():
    try:
        import mujoco
    except ImportError as e:
        help = """
        Options:
            (1) install robohive with encoders (pip install robohive['mujoco'])
            (2) follow setup instructions here: https://github.com/deepmind/mujoco
            (3) install mujoco via pip (pip install mujoco)

        """
        raise ModuleNotFoundError(f"{e}. {help}")


def dm_control_isavailable():
    try:
        import dm_control
    except ImportError as e:
        help = """
        Options:
            (1) install robohive with encoders (pip install robohive['mujoco'])
            (2) follow setup instructions here: https://github.com/deepmind/dm_control
            (3) install dm-control via pip (pip install dm-control)
        """
        raise ModuleNotFoundError(f"{e}. {help}")


def torch_isavailable():
    try:
        import torch
    except ImportError as e:
        help = """
        To use visual keys, RoboHive requires torch
        Options:
            (1) install robohive with encoders (pip install robohive['encoder'])
            (2) directly install torch via pip (pip install torch)
        """
        raise ModuleNotFoundError(f"{e}. {help}")


def torchvision_isavailable():
    try:
        import torchvision
    except ImportError as e:
        help = """
        To use visual keys, RoboHive requires torchvision
        Options:
            (1) install robohive with encoders (pip install robohive['encoder'])
            (2) directly install torchvision via pip (pip install torchvision)
        """
        raise ModuleNotFoundError(f"{e}. {help}")


def r3m_isavailable():
    try:
        import r3m
    except ImportError as e:
        help = """
        To use R3M as encodes in visual keys, RoboHive requires R3M installation
        Options:
            (1) follow install instructions at https://sites.google.com/view/robot-r3m/
            (2) pip install 'r3m@git+https://github.com/facebookresearch/r3m.git'
        """
        raise ModuleNotFoundError(f"{e}. {help}")

def vc_isavailable():
    try:
        import vc_models
    except ImportError as e:
        help = """
        To use VC1 as encodes in visual keys, RoboHive requires VC1 installation
        Options:
            (1) follow install instructions at https://eai-vc.github.io/
            (2) pip install 'vc_models@git+https://github.com/facebookresearch/eai-vc.git@9958b278666bcbde193d665cc0df9ccddcdb8a5a#egg=vc_models&subdirectory=vc_models'
        """
        raise ModuleNotFoundError(f"{e}. {help}")