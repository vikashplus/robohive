import collections
import gym
import numpy as np

from mj_envs.envs import env_base

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
        "pose": 0.0,  # 0.01
        "approach": 0.5,
    }

    def __init__(self, model_path, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path)

        self._setup(**kwargs)


    def _setup(self,
               robot_jnt_names,
               obj_jnt_names,
               obj_interaction_site,
               goal,
               interact_site,
               obj_init,
               obs_keys_wt=list(DEFAULT_OBS_KEYS_AND_WEIGHTS.keys()),
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               # different defaults than what is used in env_base and robot
               frame_skip=40,
               obs_range=(-8, 8),
               act_mode="vel",
               robot_name="Franka_kitchen_sim",
               **kwargs,
    ):

        if VIZ:
            from vtils.plotting.srv_dict import srv_dict

            self.dict_plot = srv_dict()

        # configure env-site
        self.grasp_sid = self.sim.model.site_name2id("end_effector")

        # configure env-robot
        self.robot_dofs = []
        self.robot_ranges = []
        for jnt_name in robot_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.robot_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.robot_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.robot_dofs = np.array(self.robot_dofs)
        self.robot_ranges = np.array(self.robot_ranges)
        self.robot_meanpos = np.mean(self.robot_ranges, axis=1)

        # increase simulation timestep for faster experiments
        # self.sim.model.opt.timestep = 0.008

        # configure env-objs
        self.obj = {}
        obj_dof_adrs = []
        obj_dof_ranges = []
        for goal_adr, jnt_name in enumerate(obj_jnt_names):
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.obj[jnt_name] = {}
            self.obj[jnt_name]["goal_adr"] = goal_adr
            self.obj[jnt_name]["interact_sid"] = self.sim.model.site_name2id(
                obj_interaction_site[goal_adr]
            )
            self.obj[jnt_name]["dof_adr"] = self.sim.model.jnt_dofadr[jnt_id]
            obj_dof_adrs.append(self.sim.model.jnt_dofadr[jnt_id])
            obj_dof_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.obj["dof_adrs"] = np.array(obj_dof_adrs)
        self.obj["dof_ranges"] = np.array(obj_dof_ranges)
        self.obj["dof_ranges"] = (
            self.obj["dof_ranges"][:, 1] - self.obj["dof_ranges"][:, 0]
        )

        # configure env-goal
        if interact_site == "end_effector":
            print(
                "WARNING: Using the default interaction site of end-effector. \
                  If you wish to evaluate on specific tasks, you should set the interaction site correctly."
            )
        self.set_goal(goal=goal, interact_site=interact_site)

        super()._setup(obs_keys=obs_keys_wt,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       act_mode=act_mode,
                       obs_range=obs_range,
                       robot_name=robot_name,
                       **kwargs)

        # NOTE: Keyframe[1] is obtained from one of the RPL demonstrations
        self.init_qpos[:] = self.sim.model.key_qpos[1].copy()
        self.init_qvel[:] = self.sim.model.key_qvel[1].copy()
        if obj_init:
            self.set_obj_init(obj_init)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["hand_jnt"] = sim.data.qpos[self.robot_dofs].copy()
        obs_dict["objs_jnt"] = sim.data.qpos[self.obj["dof_adrs"]].copy()
        obs_dict["hand_vel"] = sim.data.qvel[self.robot_dofs].copy() * self.dt
        obs_dict["objs_vel"] = sim.data.qvel[self.obj["dof_adrs"]].copy() * self.dt
        obs_dict["goal"] = self.goal.copy()
        obs_dict["goal_err"] = (
            obs_dict["goal"] - obs_dict["objs_jnt"]
        )  # mix of translational and rotational erros
        obs_dict["approach_err"] = (
            self.sim.data.site_xpos[self.interact_sid]
            - self.sim.data.site_xpos[self.grasp_sid]
        )
        obs_dict["pose_err"] = self.robot_meanpos - obs_dict["hand_jnt"]
        obs_dict["end_effector"] = self.sim.data.site_xpos[self.grasp_sid]
        obs_dict["qpos"] = self.sim.data.qpos.copy()
        for site in self.INTERACTION_SITES:
            site_id = self.sim.model.site_name2id(site)
            obs_dict[site + "_err"] = (
                self.sim.data.site_xpos[site_id]
                - self.sim.data.site_xpos[self.grasp_sid]
            )
        return obs_dict

    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(obs_dict["goal_err"])

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("goal", -np.sum(goal_dist, axis=-1)),
                (
                    "bonus",
                    np.product(goal_dist < 0.75 * self.obj["dof_ranges"], axis=-1)
                    + np.product(goal_dist < 0.25 * self.obj["dof_ranges"], axis=-1),
                ),
                ("pose", -np.sum(np.abs(obs_dict["pose_err"]), axis=-1)),
                ("approach", -np.linalg.norm(obs_dict["approach_err"], axis=-1)),
                # Must keys
                ("sparse", -np.sum(goal_dist, axis=-1)),
                ("solved", np.all(goal_dist < 0.15 * self.obj["dof_ranges"])),
                ("done", False),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        if self.mujoco_render_frames and VIZ:
            self.dict_plot.append(rwd_dict, self.rwd_keys_wt)
            # self.dict_plot.append(rwd_dict)

        return rwd_dict

    # Set object initializations
    # Note: this doesn't directly initializes the objects. One has to either call reset or explicitely use init_qpos manually set the sim.
    def set_obj_init(self, obj_init):
        # resolve goals
        if type(obj_init) is dict:
            # overwrite explicit requests
            for obj_name, obj_val in obj_init.items():
                self.init_qpos[self.obj[obj_name]["dof_adr"]] = obj_val
        elif type(obj_init) is np.ndarray:
            assert len(obj_init) == len(
                self.obj["dof_adrs"]
            ), "Check size of provided onj_init"
            self.init_qpos[self.obj["dof_adrs"]] = obj_init.copy()
        else:
            raise TypeError(
                "obj_init must be either a dict<obj_name, obb_init>, or a vector of all obj_init"
            )

    def set_goal(self, goal=None, interact_site=None):

        # resolve goals
        if type(goal) is dict:
            # treat current sim as goal
            self.goal = self.sim.data.qpos[self.obj["dof_adrs"]].copy()
            # overwrite explicit requests
            for obj_name, obj_goal in goal.items():
                self.goal[self.obj[obj_name]["goal_adr"]] = obj_goal
        elif type(goal) is np.ndarray:
            assert len(goal) == len(self.obj["dof_adrs"]), "Check size of provided goal"
            self.goal = goal
        else:
            raise TypeError(
                "goals must be either a dict<obj_name, obb_goal>, or a vector of all obj_goals"
            )

        # resolve interaction site
        if interact_site is None:  # automatically infer
            goal_err = np.abs(self.sim.data.qpos[self.obj["dof_adrs"]] - self.goal)
            max_goal_err_obj = np.argmax(goal_err)
            for _, obj in self.obj.items():
                if obj["goal_adr"] == max_goal_err_obj:
                    self.interact_sid = obj["interact_sid"]
                    break
        elif type(interact_site) is str:  # overwrite using name
            self.interact_sid = self.sim.model.site_name2id(interact_site)
        elif type(interact_site) is int:  # overwrite using id
            self.interact_sid = interact_site


class KitchenFrankaFixed(KitchenBase):

    INTERACTION_SITES = (
        "knob1_site",
        "knob2_site",
        "knob3_site",
        "knob4_site",
        "light_site",
        "slide_site",
        "leftdoor_site",
        "rightdoor_site",
        "microhandle_site",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
    )

    OBJ_JNT_NAMES = (
        "knob1_joint",
        "knob2_joint",
        "knob3_joint",
        "knob4_joint",
        "lightswitch_joint",
        "slidedoor_joint",
        "leftdoorhinge",
        "rightdoorhinge",
        "microjoint",
        "kettle0:Tx",
        "kettle0:Ty",
        "kettle0:Tz",
        "kettle0:Rx",
        "kettle0:Ry",
        "kettle0:Rz",
    )

    ROBOT_JNT_NAMES = (
        "panda0_joint1",
        "panda0_joint2",
        "panda0_joint3",
        "panda0_joint4",
        "panda0_joint5",
        "panda0_joint6",
        "panda0_joint7",
        "panda0_finger_joint1",
        "panda0_finger_joint2",
    )

    def __init__(
        self,
        model_path,
        robot_jnt_names=ROBOT_JNT_NAMES,
        obj_jnt_names=OBJ_JNT_NAMES,
        obj_interaction_site=INTERACTION_SITES,
        goal=None,
        interact_site="end_effector",
        obj_init=None,
        **kwargs,
    ):
        KitchenBase.__init__(
            self,
            model_path=model_path,
            robot_jnt_names=robot_jnt_names,
            obj_jnt_names=obj_jnt_names,
            obj_interaction_site=obj_interaction_site,
            goal=goal,
            interact_site=interact_site,
            obj_init=obj_init,
            **kwargs,
        )


class KitchenFrankaDemo(KitchenFrankaFixed):
    def reset(self, *args, **kwargs):
        reset_qpos = self.sim.model.key_qpos[1].copy()
        reset_qvel = self.sim.model.key_qvel[1].copy()
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)


class KitchenFrankaRandom(KitchenFrankaFixed):
    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)


class KitchenFrankaRandomDesk(KitchenFrankaFixed):
    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
            ncon = 10
            while ncon > 4:
                self.sim.model.body_pos[14] = np.array([-0.1, 0.75, 0.0]) + np.random.uniform(-0.1, 0.1, (3,))
                self.sim.step()
                ncon = self.sim.data.ncon
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)
