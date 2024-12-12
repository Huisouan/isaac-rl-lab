from omni.isaac.lab.utils import configclass

from .rough_env_cfg import UnitreeA1HimRoughEnvCfg, UnitreeA1RoughEnvCfg_PLAY


@configclass
class UnitreeA1HimFlatEnvCfg(UnitreeA1HimRoughEnvCfg):
    def __post_init__(self):
        # Temporarily not run disable_zerow_eight_rewards() in parent class to override rewards
        self._run_disable_zero_weight_rewards = False
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        #if flat ,use flat base height reward
        self.rewards.base_height_rough_l2.weight =  0
        self.rewards.base_height_rough_l2.params["target_height"] = 0.4        
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.4       
        # Now executing disable_zerow_eight_rewards()
        self._run_disable_zero_weight_rewards = True
        if self._run_disable_zero_weight_rewards:
            self.disable_zero_weight_rewards()
