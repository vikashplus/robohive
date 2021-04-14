from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os
import numpy as np
import vtils.plotting.simple_plot as plt

# user inputs ===========================
# MODEL_PATH = "hand/Index_Thumb_v0.xml"
MODEL_PATH = "basic/muscle_load.xml"
HORIZON = 1
ACTIVATION_AMP = 1
ACTIVATION_FRQ = 2
RENDER = False

model = load_model_from_path(MODEL_PATH)
sim = MjSim(model)
if RENDER:
    viewer = MjViewer(sim)
t = 0

trace = {}
trace['qpos'] = []
trace['qvel'] = []
trace['ctrl'] = []
trace['acti'] = []
trace['forc'] = []

while sim.data.time<HORIZON:
    t = sim.data.time
    sim.data.ctrl[0] = math.cos(t * ACTIVATION_FRQ * 2 * np.pi) * ACTIVATION_AMP
    # sim.data.ctrl[1] = math.sin(t / 10.) * 1.01

    if t>.25 and t <.30:
        sim.data.ctrl[0] = 1
    else:
        sim.data.ctrl[0] = 0

    trace['qpos'].append(sim.data.qpos.copy())
    trace['qvel'].append(sim.data.qvel.copy())
    trace['ctrl'].append(sim.data.ctrl.copy())
    trace['acti'].append(sim.data.act.copy())
    trace['forc'].append(sim.data.actuator_force.copy())
    sim.step()
    if RENDER:
        viewer.render()

if not RENDER:
    plt.plot(xdata=trace['ctrl'], subplot_id=(2, 3, 1), fig_name="muscle probe", plot_name="spikes/actions/controls")

    plt.plot(xdata=trace['qpos'], subplot_id=(2, 3, 2), fig_name="muscle probe", plot_name="qpos")
    plt.plot(xdata=trace['qvel'], subplot_id=(2, 3, 3), fig_name="muscle probe", plot_name="qvel")
    plt.plot(xdata=trace['acti'], subplot_id=(2, 3, 5), fig_name="muscle probe", plot_name="muscle activations")
    plt.plot(xdata=trace['forc'], subplot_id=(2, 3, 6), fig_name="muscle probe", plot_name="muscle force")
    plt.show_plot()
# import ipdb; ipdb.set_trace()
