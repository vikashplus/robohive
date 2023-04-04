# Frequently Asked Questions

## Visualization faq
1. If the visualization results in a GLFW error, this is because `mujoco-py` does not see some graphics drivers correctly. This can usually be fixed by explicitly loading the correct drivers before running the python script. See [this page](https://github.com/aravindr93/mjrl/tree/master/setup#known-issues) for details.
2. If FFmpeg isn't found then run `apt-get install ffmpeg` on linux and `brew install ffmpeg` on osx (`conda install FFmpeg` causes some issues)
