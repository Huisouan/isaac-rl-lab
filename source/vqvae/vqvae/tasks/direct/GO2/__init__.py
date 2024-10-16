# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .go2_pmc_env import PMCEnvCfg,PMCEnv
from .go2_epmc_env import EPMCEnvCfg,EPMCEnv
##
# Register Gym environments.
##

gym.register(
    id="Isaac-go2-pmc-Direct-v0",
    entry_point="vqvae.tasks.direct.GO2.go2_pmc_env:PMCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PMCEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",

    },
)

gym.register(
    id = "Isaac-go2-epmc-Direct-v0",
    entry_point="vqvae.tasks.direct.GO2.go2_pmc_env:PMCEnv",    
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PMCEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",

    },
)