"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_reward_weight(env: RLTaskEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight after some number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_push_force(
        env: RLTaskEnv, 
        env_ids: Sequence[int], 
        term_name: str, 
        max_velocity: Sequence[float], 
        interval: int, 
        starting_step: float = 0.0
        ):
    """Curriculum that modifies the maximum push (perturbation) velocity over some intervals. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity of the push.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """
    try:
        term_cfg = env.event_manager.get_term_cfg(term_name)
    except Exception:
        # print("No push_robot term found in the event manager")
        return 0.0
    
    curr_x = float(term_cfg.params["velocity_range"]["x"][1])
    curr_y = float(term_cfg.params["velocity_range"]["y"][1])

    if env.common_step_counter < starting_step:
        return curr_x
    
    if env.common_step_counter % interval == 0:
        base_contact = torch.sum(env.termination_manager.get_term("base_contact"))
        time_out = torch.sum(env.termination_manager.get_term("time_out"))

        if time_out > 0 and base_contact < time_out * 0.5:
            curr_x = float(np.clip(curr_x + 0.1, 0.0, max_velocity[0]))
            curr_y = float(np.clip(curr_y + 0.1, 0.0, max_velocity[1]))
            term_cfg.params["velocity_range"]["x"] = (-curr_x, curr_x)
            term_cfg.params["velocity_range"]["y"] = (-curr_y, curr_y)
            env.event_manager.set_term_cfg(term_name, term_cfg)
        elif base_contact > time_out:
            curr_x = float(np.clip(curr_x - 0.1, 0.0, max_velocity[0]))
            curr_y = float(np.clip(curr_y - 0.1, 0.0, max_velocity[1]))
            term_cfg.params["velocity_range"]["x"] = (-curr_x, curr_x)
            term_cfg.params["velocity_range"]["y"] = (-curr_y, curr_y)
            env.event_manager.set_term_cfg(term_name, term_cfg)

    return curr_x


def modify_command_velocity(
    env: RLTaskEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    max_velocity: Sequence[float], 
    interval: int, 
    starting_step: float = 0.0
    ):
    """Curriculum that modifies the maximum command velocity over some intervals. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity. 
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """

    command_cfg = env.command_manager.get_term('base_velocity').cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x

    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]
    
    if env.common_step_counter % interval == 0 and len(env_ids) > 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        episode_sums = env.reward_manager._episode_sums
        if isinstance(episode_sums, dict):
            rew = episode_sums[term_name][env_ids]
        else:
            term_idx = env.reward_manager.active_terms.index(term_name)
            rew = (
                episode_sums[env_ids, term_idx]
                if episode_sums.shape[0] == env.num_envs
                else episode_sums[term_idx, env_ids]
            )

        if torch.mean(rew) / env.max_episode_length > 0.8 * term_cfg.weight * env.step_dt:
            curr_lin_vel_x = (
                float(np.clip(curr_lin_vel_x[0] - 0.2, max_velocity[0], 0.0)),
                float(np.clip(curr_lin_vel_x[1] + 0.2, 0.0, max_velocity[1])),
            )
            command_cfg.ranges.lin_vel_x = curr_lin_vel_x

    return curr_lin_vel_x[1]