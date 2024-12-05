from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import Camera, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def extent_force(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the external forces and torques applied to the bodies.
    This function returns the external forces and torques applied to the bodies. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # Assuming body_ids is a single-element list or array
    return torch.cat((asset._external_force_b[:, body_id, :], asset._external_torque_b[:, body_id, :]), dim=1)