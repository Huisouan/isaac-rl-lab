#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1
STYLE_INIT_SCALE = 1.0
class ASEV1(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        amp_obs,
        num_envs,
        ase_latent_shape = 64,
        
        actor_hidden_dims=[1024, 1024, 512, 12],
        critic_hidden_dims=[1024, 1024, 512, 1],
        disc_hidden_dims=[1024, 1024, 512],
        enc_hidden_dims=[1024, 512],
        stylenet_hedden_dims=[512, 256],
        activation="relu",
        init_noise_std=1.0,

        latent_steps_min:int =  1,
        latent_steps_max:int =  150    ,        
        
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)
        stylenet_act = get_activation("tanh")
        mlp_input_dim_a = num_actor_obs+ase_latent_shape
        mlp_input_dim_v = num_critic_obs+ase_latent_shape
        mlp_input_dim_c = mlp_input_dim_e = amp_obs
        # 将actor_hidden_dims的最后一个元素设置为num_actions
        actor_hidden_dims[-1] = num_actions
        
        #init params
        self.latent_steps_min = latent_steps_min
        self.latent_steps_max = latent_steps_max

        self.ase_latent_shape = ase_latent_shape
        env_ids = torch.tensor(np.arange(num_envs), dtype=torch.long)
        self.ase_latents = self.sample_latents(num_envs)
        self.latent_reset_steps = torch.zeros(num_envs, dtype=torch.int32)
        self.reset_latent_step_count()
        #Actor
        self.style_net = create_mlp(ase_latent_shape, stylenet_hedden_dims, activation,activation)
        self.style_net_out = create_mlp(stylenet_hedden_dims[-1], [ase_latent_shape], stylenet_act)
        self.actor = create_mlp(mlp_input_dim_a, actor_hidden_dims, activation)
        #Critic
        self.critic = create_mlp(mlp_input_dim_v, critic_hidden_dims, activation)
        #Discriminator
        self.disc = create_mlp(mlp_input_dim_c, disc_hidden_dims, activation,activation)
        self.disc_logits = nn.Linear(disc_hidden_dims[-1], 1)
        #Encoder
        self.enc = create_mlp(mlp_input_dim_e, enc_hidden_dims, activation,activation)
        self.enc_logits = nn.Linear(enc_hidden_dims[-1], ase_latent_shape)
        
        print(f"Style MLP: {self.style_net}")
        print(f"Style Out : {self.style_net_out}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Disc MLP: {self.disc}")
        print(f"Disc Logits : {self.disc_logits}")
        print(f"Enc MLP: {self.enc}")
        print(f"Enc Out : {self.enc_logits}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        for m in self.modules():         
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)  # 初始化MLP偏置  
                            
        for m in self.style_net_out.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -STYLE_INIT_SCALE,STYLE_INIT_SCALE)         
        torch.nn.init.uniform_(self.disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.uniform_(self.enc_logits.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)
    

    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        pass

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset_latent_step_count(self, env_ids):
        # 为指定环境ID重置潜在步数计数
        self.latent_reset_steps[env_ids] = torch.randint_like(
            self.latent_reset_steps[env_ids], low=self.latent_steps_min, high=self.latent_steps_max)
    
    def sample_latents(self, n):
        z = torch.normal(torch.zeros([n, self.ase_latent_shape]))  # 生成正态分布的潜在变量
        z = torch.nn.functional.normalize(z, dim=-1)  # 归一化潜在变量
        return z

    def update_distribution(self, observations):
        # Check for NaN values in the observations tensor
        # Compute the mean using the actor network
        mean = self.actor(observations)

        # Check for NaN values in the mean tensor
        if torch.isnan(mean).any():
            print(f"NaN detected in mean tensor: {mean}")
            raise ValueError("Mean computed by actor contains NaN values")

        # Update the distribution
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

def create_mlp(input_dim, hidden_dims, activation, output_activation=None):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation)
    for layer_index in range(1, len(hidden_dims)):
        if layer_index == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[layer_index - 1], hidden_dims[layer_index]))
        else:
            layers.append(nn.Linear(hidden_dims[layer_index - 1], hidden_dims[layer_index]))
            layers.append(activation)
        
    # 最后一层的输出维度与 hidden_dims 的最后一层相同
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)
