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
from .go2_amp_env import AMPEnvCfg,AMPEnv
##
# Register Gym environments.
##

#vqvae
gym.register(
    id="Isaac-go2-pmc-Direct-v0",
    entry_point="rl_lab.tasks.direct.GO2.go2_pmc_env:PMCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PMCEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPMCPPORunnerCfg",
    },
)
#cvqvae，相比vqvae只修改了码本的更新方式，因此使用与vqvae相同的环境注册方式
gym.register(
    id="Isaac-go2-cvqvae-Direct-v0",
    entry_point="rl_lab.tasks.direct.GO2.go2_pmc_env:PMCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PMCEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughCVQVAEPPORunnerCfg",
    },
)
#amp 不使用这个注册
gym.register(
    id="Isaac-go2-amp-Direct-v0",
    entry_point="rl_lab.tasks.direct.GO2.go2_amp_env:AMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AMPEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatAMPPPORunnerCfg",
    },
)
