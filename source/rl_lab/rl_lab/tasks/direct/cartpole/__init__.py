# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .cartpole_camera_env import CartpoleCameraEnv, CartpoleDepthCameraEnvCfg, CartpoleRGBCameraEnvCfg
from .cartpole_env import CartpoleEnv, CartpoleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-pmc-v0",
    entry_point="vqvae.tasks.direct.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
    },
)

