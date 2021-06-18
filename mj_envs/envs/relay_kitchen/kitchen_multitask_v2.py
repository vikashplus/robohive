import numpy as np
from mj_envs.envs import env_base
import os
import collections

VIZ = False

class KitchenBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS_AND_WEIGHTS = {
        "hand_jnt": 1.0,
        "objs_jnt": 1.0,
        "goal": 1.0,
        "goal_err": 1.0,
        "approach_err": 1.0,
    }
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal": 1.0,
        "bonus": 0.5,
        "pose": 0*0.01,
        "approach": 0.5,
    }

    def __init__(self, model_path, config_path,
                    robot_jnt_names, obj_jnt_names, goal, interact_site,
                    obs_keys_wt = DEFAULT_OBS_KEYS_AND_WEIGHTS,
                    rwd_keys_wt = DEFAULT_RWD_KEYS_AND_WEIGHTS,
                    **kwargs):

        if VIZ:
            from vtils.plotting.srv_dict import srv_dict
            self.dict_plot = srv_dict()

        # get sims
        self.sim = env_base.get_sim(model_path=model_path)
        self.sim_obsd = env_base.get_sim(model_path=model_path)

        # configure env-site
        self.grasp_sid = self.sim.model.site_name2id('end_effector')
        self.interact_sid = self.sim.model.site_name2id(interact_site)

        # configure env-robot
        self.robot_dofs = []
        self.robot_ranges = []
        for jnt_name in robot_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.robot_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.robot_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.robot_dofs = np.array(self.robot_dofs)
        self.robot_ranges = np.array(self.robot_ranges)
        self.robot_meanpos =  np.mean(self.robot_ranges,axis=1)

        # configure env-objs
        self.obj_dofs = []
        self.obj_ranges = []
        for jnt_name in obj_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.obj_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.obj_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.obj_dofs = np.array(self.obj_dofs)
        self.obj_ranges = np.array(self.obj_ranges)
        self.obj_ranges = self.obj_ranges[:,1] - self.obj_ranges[:,0]

        # configure env-goal
        self.set_goal(goal)

        # get env
        env_base.MujocoEnv.__init__(self,
                                sim = self.sim,
                                sim_obsd = self.sim_obsd,
                                frame_skip = 40,
                                config_path = config_path,
                                obs_keys = list(obs_keys_wt.keys()),
                                rwd_keys_wt = rwd_keys_wt,
                                rwd_mode = "dense",
                                act_mode = "vel",
                                act_normalized = True,
                                is_hardware = False,
                                robot_name = "Franka_kitchen_sim",
                                obs_range = (-8, 8),
                                **kwargs)

        self.init_qpos = self.sim.model.key_qpos[0].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[self.robot_dofs].copy()
        obs_dict['objs_jnt'] = sim.data.qpos[self.obj_dofs].copy()
        obs_dict['hand_vel'] = sim.data.qvel[self.robot_dofs].copy() * self.dt
        obs_dict['objs_vel'] = sim.data.qvel[self.obj_dofs].copy() * self.dt
        obs_dict['goal'] = self.goal.copy()
        obs_dict['goal_err'] = obs_dict['goal']-obs_dict['objs_jnt'] #??? Kettle has quaternions
        obs_dict['approach_err'] = self.sim.data.site_xpos[self.interact_sid] - self.sim.data.site_xpos[self.grasp_sid]
        obs_dict['pose_err'] = self.robot_meanpos-obs_dict['hand_jnt']
        return obs_dict


    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(obs_dict['goal_err'])

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('goal',    -np.sum(goal_dist, axis=-1)),
            ('bonus',   np.sum(goal_dist < 0.75*self.obj_ranges, axis=-1) + np.sum(goal_dist < 0.25*self.obj_ranges, axis=-1)),
            ('pose',    -np.sum(np.abs(obs_dict['pose_err']), axis=-1)),
            ('approach',-np.linalg.norm(obs_dict['approach_err'], axis=-1)),
            # Must keys
            ('sparse',  -np.sum(goal_dist, axis=-1)),
            ('solved',  np.all(goal_dist < 0.15*self.obj_ranges)),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        if self.mujoco_render_frames and VIZ:
            self.dict_plot.append(rwd_dict, self.rwd_keys_wt)
            # self.dict_plot.append(rwd_dict)

        return rwd_dict


    def set_goal(self, goal=None):
        if goal is not None:
            assert len(goal) == len(self.obj_dofs), "Check size of provided goal"
            self.goal = goal
        else: # treat current sim as goal
            self.goal = self.sim.data.qpos[self.obj_dofs].copy()


class KitchenFrankaFixed(KitchenBase):

    def __init__(self,
                goal=None,
                interact_site="end_effector",
                **kwargs,
                ):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        KitchenBase.__init__(self,
            model_path = curr_dir + '/assets/franka_kitchen.xml',
            config_path = curr_dir + '/assets/franka_kitchen.config',
            robot_jnt_names = ('panda0_joint1', 'panda0_joint2', 'panda0_joint3', 'panda0_joint4', 'panda0_joint5', 'panda0_joint6', 'panda0_joint7', 'panda0_finger_joint1', 'panda0_finger_joint2'),
            obj_jnt_names = ('knob_Joint_1', 'knob_Joint_2', 'knob_Joint_3', 'knob_Joint_4', 'lightswitch_joint', 'slidedoor_joint', 'leftdoorhinge', 'rightdoorhinge', 'microjoint'),
            goal=goal,
            interact_site=interact_site,
            **kwargs)


class KitchenFrankaRandom(KitchenFrankaFixed):
    def reset(self, reset_qpos=None, reset_qvel=None):
        init_qpos_perturbed = self.init_qpos.copy()
        init_qpos_perturbed[self.robot_dofs] += 0.05*(self.np_random.uniform(size=len(self.robot_dofs))-0.5)*(self.robot_ranges[:,1] - self.robot_ranges[:,0])
        return super().reset(reset_qpos=init_qpos_perturbed)
