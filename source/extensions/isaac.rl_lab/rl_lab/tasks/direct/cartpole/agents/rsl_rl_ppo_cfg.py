# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .....tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoPMCCfg,
    Z_settings,    
    
)


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 9000
    save_interval = 50
    experiment_name = "CartpolePPORunner"
    empirical_normalization = False
    policy = RslRlPpoPMCCfg(
        class_name="PMC",
        init_noise_std=0.1,
        actor_hidden_dims=[256, 256],
        encoder_hidden_dims=[256, 256],
        decoder_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="relu",
        State_Dimentions = 4*3
        
    )
    z_settings = Z_settings(
            z_length = 32,
            num_embeddings = 256,
            norm_z = False,
            bot_neck_z_embed_size = 32,
            bot_neck_prop_embed_size = 64,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.00,
        num_learning_epochs=16,
        num_mini_batches=4,
        learning_rate=0.00001,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )

