from __future__ import annotations

import torch

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_rotate

from robot_lab.tasks.locomotion.velocity.config.unitree_a1_amp.env.manager_based_rl_amp_env import ManagerBasedRLAmpEnv


def reset_amp(
    env: ManagerBasedRLAmpEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    return 0