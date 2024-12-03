import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Amp-Unitree-go2-v0",
    entry_point="rl_lab.tasks.manager_based.locomotion.velocity.config.go2_amp.env.manager_based_rl_amp_env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeA1AmpFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1AmpFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rough-Amp-Unitree-go2-v0",
    entry_point="rl_lab.tasks.manager_based.locomotion.velocity.config.go2_amp.env.manager_based_rl_amp_env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeA1AmpRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1AmpRoughPPORunnerCfg",
    },
)
