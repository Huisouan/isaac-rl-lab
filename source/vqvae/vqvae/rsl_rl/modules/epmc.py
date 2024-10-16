#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from ...tasks.utils.wappers.rsl_rl import (
    Z_settings,
)





class PMC(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        activation="relu",
        init_noise_std=-2.0,
        **kwargs,
    ):
        # 检查是否有未使用的参数
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
 
        #######################################Value#################################################################
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder},Codebook :{self.codebook},Decoder MLP: {self.decoder}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.z_e = None
        self.z_q = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    @property
    def vector_z_e(self):
        return self.z_e

    @property
    def vector_z_q(self):
        return self.z_q
    @property
    def encode_one_hot(self):
        return self.one_hot
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        observations = self.rms(observations)
        
        self.z_e = self.encoder(observations)
        
        # 将潜空间向量与codebook求最近距离，得到量化后的向量
        flat_inputs = self.z_e.view(-1, self.z_length)
        embeddings = self.get_codebook_embeddings()
        nearest = torch.argmin(torch.cdist(flat_inputs, embeddings), dim=1)
        self.z_q = self.codebook(nearest)
        self.one_hot = F.one_hot(nearest, num_classes=self.num_embeddings).float()
        
        #将量化后的向量与观测值进行拼接
        decoder_input = torch.cat((self.z_embd(self.z_e + (self.z_q - self.z_e.detach())),self.observation_embd(observations[:, :self.State_Dimentions])), dim=1)
        
        #计算输出动作
        mean = self.decoder(decoder_input)
        
        #Update State Memory
        #self.state_t_minus2 = self.state_t_minus1
        #self.state_t_minus1 = self.state_t
        
        # 使用均值和标准差创建一个正态分布对象
        # 其中标准差为均值乘以0（即不改变均值）再加上self.std
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return mean
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_codebook_embeddings(self):
        return self.codebook.weight    

    def act_inference(self, observations):
        actions_mean = self.update_distribution(observations)
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
