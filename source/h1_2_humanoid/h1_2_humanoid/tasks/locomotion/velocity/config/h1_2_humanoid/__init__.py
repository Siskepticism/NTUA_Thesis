

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Flat-H1_2-Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
       "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HumanoidFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:HumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-H1_2-Humanoid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HumanoidFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:HumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-H1_2-Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HumanoidRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:HumanoidRoughPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-H1_2-Humanoid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
       "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HumanoidRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:HumanoidRoughPPORunnerCfg",
    },
)