
from omni.isaac.lab.utils import configclass

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
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -1.0

        self.observations.policy.height_scan = None  #because we have proprioception only
        self.scene.height_scanner = None

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