# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """基于机器人在被命令以期望速度移动时行走的距离的课程设置。

    该术语用于在机器人行走足够远时增加地形的难度，并在机器人行走距离少于所需距离的一半时降低难度。

    .. note::
        仅当使用 ``generator`` 类型的地形时才能使用此术语。有关不同地形类型的更多信息，请参阅 :class:`omni.isaac.lab.terrains.TerrainImporter` 类。

    返回:
        给定环境ID的平均地形级别。
    """
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # 计算机器人行走的距离
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 行走足够远的机器人将进入更难的地形
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 行走距离少于所需距离一半的机器人将进入更简单的地形
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # 更新地形级别
    terrain.update_env_origins(env_ids, move_up, move_down)
    # 返回平均地形级别
    return torch.mean(terrain.terrain_levels.float())
