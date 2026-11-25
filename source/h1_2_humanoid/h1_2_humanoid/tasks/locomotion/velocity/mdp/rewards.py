from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def feet_air_time(env: RLTaskEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold_min: float,
                  threshold_max: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    cmd = env.command_manager.get_command(command_name)
    reward *= (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold_min: float, threshold_max: float,
                                 sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold_max)
    # no reward for small steps
    reward *= reward > threshold_min
    # no reward for zero command
    cmd = env.command_manager.get_command(command_name)
    reward *= (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_min_distance(env:RLTaskEnv, min_dist:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll_link")) -> torch.Tensor:
    """ Penalizing when feet get closer (in XY) to each other so they dont touch"""
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :2]
    dist = torch.norm(feet_pos[:,0,:] - feet_pos[:,1,:], dim=-1)
    reward = torch.clamp(min_dist - dist, min = 0.0)
    return reward

def feet_lateral_distance(env:RLTaskEnv, min_dist:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll_link")) -> torch.Tensor:
    """ Penalize  when feet get closerthan min_dist in lateral axis"""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :3]
    delta_w = feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :]
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), delta_w)
    lateral_dist = torch.abs(delta_b[:, 1])
    reward = torch.clamp(min_dist - lateral_dist, min=0.0)
    return reward

def feet_clearance_l2(
    env: RLTaskEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll_link")) -> torch.Tensor:
    """Penalize deviation from target swing height weighted by horizontal foot speed."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    v = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    reward = ((z - target_height) ** 2 * torch.tanh(2.0 * v)).sum(dim=1)
    return reward


def feet_clearance_relative(
    env: RLTaskEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll_link")) -> torch.Tensor:
    """Penalize deviation from target swing height relative to lower foot (meaning real - terrain height)."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    rel = z - z.min(dim=1, keepdim=True)[0]
    v = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    reward = ((rel - target_height) ** 2 * torch.tanh(2.0 * v)).sum(dim=1)
    return reward