
from isaaclab.utils import configclass
import h1_2_humanoid.tasks.locomotion.velocity.mdp as mdp
from h1_2_humanoid.tasks.locomotion.velocity.velocity_env_cfg import \
    LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from h1_2_humanoid.assets.h1_2 import H1_2_HUMANOID_CFG


@configclass
class HumanoidRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = H1_2_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        #Z-axis penalties (not needed when using proprioception/not height-scanners)
        self.rewards.base_height_l2.weight = 0.0

        self.rewards.feet_clearance.func = mdp.feet_clearance_relative
        self.rewards.feet_clearance.weight = -0.75
        self.rewards.feet_clearance.params["target_height"] = 0.08

        #Rough Terrain rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.joint_deviation_hip.weight = -0.05
        self.rewards.joint_deviation_knee.weight = -0.02

        self.observations.policy.height_scan = None  #because we have proprioception only
        self.observations.policy.history_length = 5


        
        #spawn straight        
        self.randomization.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.curriculum.push_force_levels.params["max_velocity"] = [1.5, 1.5]
        self.curriculum.command_vel.params["max_velocity"] = [-0.8, 1.5]

@configclass
class HumanoidRoughEnvCfg_PLAY(HumanoidRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None