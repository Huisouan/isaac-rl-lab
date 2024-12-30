import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ase-Unitree-go2-v0",
    entry_point="rl_lab.tasks.manager_based.locomotion.velocity.config.unitree_go2_ase.env.manager_based_rl_amp_env:ManagerBasedRLASEEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeGO2ASEFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGO2ASEFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rough-Ase-Unitree-go2-v0",
    entry_point="rl_lab.tasks.manager_based.locomotion.velocity.config.unitree_go2_ase.env.manager_based_rl_amp_env:ManagerBasedRLASEEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeGO2ASERoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGO2ASERoughPPORunnerCfg",
    },
)
