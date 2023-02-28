from termcolor import cprint
import enum

class Prompt(enum.IntEnum):
    """Prompt verbosity types"""
    INFO = 0
    WARN = 1
    ERROR = 2
    NONE = 4

# Global verbose mode
VERBOSE_MODE = Prompt.INFO

def set_prompt_verbosity(verbose_mode:Prompt=Prompt.INFO):
    global VERBOSE_MODE
    VERBOSE_MODE = verbose_mode


def prompt(data, color=None, on_color=None, flush=False, end="\n", type:Prompt=Prompt.INFO):
    if type>=VERBOSE_MODE:
        cprint(data, color=color, on_color=on_color, flush=flush, end=end)
