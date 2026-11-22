
from isaaclab.utils import configclass

from .rough_env_cfg import HumanoidRoughEnvCfg


@configclass
class HumanoidFlatEnvCfg(HumanoidRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold_min"] = 0.2
        self.rewards.feet_air_time.params["threshold_max"] = 0.4
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        #spawn straight        
        self.randomization.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.push_force_levels = None
        self.curriculum.command_vel = None

@configclass
class HumanoidFlatEnvCfg_PLAY(HumanoidFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None