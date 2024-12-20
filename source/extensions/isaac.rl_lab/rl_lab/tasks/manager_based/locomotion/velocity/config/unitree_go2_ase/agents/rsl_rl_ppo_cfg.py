from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from rl_lab.tasks.utils.wrappers.rsl_rl import (
    ASECfg,ASENetcfg
)

@configclass
class UnitreeA1AmpRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "unitree_a1_ase_rough"
    empirical_normalization = False

    config = ASECfg()
    
    asenetcfg = ASENetcfg()
    
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="ASEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    amp_reward_coef = 2.0
    amp_task_reward_lerp = 0.5
    amp_discr_hidden_dims = [1024, 512]
    min_normalized_std = [0.01, 0.01, 0.01] * 4


@configclass
class UnitreeA1AmpFlatPPORunnerCfg(UnitreeA1AmpRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.max_iterations = 300
        self.experiment_name = "unitree_a1_ase_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]
