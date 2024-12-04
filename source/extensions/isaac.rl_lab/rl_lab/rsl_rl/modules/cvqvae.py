#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import RMSNorm
from .Cvqvae.quantise import VectorQuantiser

from rl_lab.tasks.utils.wappers.rsl_rl import (
    Z_settings,
)




class CVQVAE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        encoder_hidden_dims=[256, 256],
        decoder_hidden_dims=[256, 256],
        critic_hidden_dims=[256,256],
        activation="lrelu",
        value_activation="tanh",
        init_noise_std=1,
        # 正确的方式是先声明类型，然后创建对象
        z_settings: Z_settings = Z_settings(),  # 将类型注释和对象创建分开
        State_Dimentions = 45*3,
        rms_momentum = 0.0001,
        **kwargs,
    ):
        # 检查是否有未使用的参数
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)
        value_activation = get_activation(value_activation)
        
        
        self.z_length = z_settings.z_length
        self.input_init = False
        self.State_Dimentions = State_Dimentions
        self.num_embeddings = z_settings.num_embeddings
        #Decoder 
        mlp_input_dim_c = num_critic_obs
        #######################################RMS#################################################################
        rms_layers = []
        rms_layers.append(RMSNorm(num_actor_obs, eps=1e-6))
        self.rms = nn.Sequential(*rms_layers)

        #######################################Actor#################################################################
        #VQVAE Encoder
        encoder_input = num_actor_obs
        encoder_output = self.z_length
        encoer_layers = []
        
        encoer_layers.append(nn.Linear(encoder_input, encoder_hidden_dims[0]))
        encoer_layers.append(activation)
        for layer_index in range(len(encoder_hidden_dims)):
            if layer_index == len(encoder_hidden_dims) - 1:
                # output layer
                encoer_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_output))
            else:
                # hidden layer
                encoer_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
                encoer_layers.append(activation)

        
        #encoderlayer
        self.encoder = nn.Sequential(*encoer_layers)
        

        #codebook
        """
        self.codebook = nn.Embedding(self.num_embeddings, encoder_output)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / z_settings.num_embeddings, 1.0 / encoder_output
        )
        """
        self.codebook = VectorQuantiser(self.num_embeddings, encoder_output,beta = 0.25)
        
        # separate embed layer 
        observation_embd_layers = []
        observation_embd_layers.append(nn.Linear(State_Dimentions,z_settings.bot_neck_prop_embed_size))
        observation_embd_layers.append(activation)
        self.observation_embd = nn.Sequential(*observation_embd_layers)
        
        z_embd_layers = []
        z_embd_layers.append(nn.Linear(encoder_output, z_settings.bot_neck_z_embed_size))
        z_embd_layers.append(activation)
        self.z_embd = nn.Sequential(*z_embd_layers)
        

        #VQVAE Decoder
        #这里encoder的输入层是隐空间向量与当前观测状态的拼接
        decoder_input = z_settings.bot_neck_prop_embed_size+z_settings.bot_neck_z_embed_size
        decoder_output = num_actions
        decoer_layers = []
        
        decoer_layers.append(nn.Linear(decoder_input, decoder_hidden_dims[0]))
        decoer_layers.append(activation)
        for layer_index in range(len(decoder_hidden_dims)):
            if layer_index == len(decoder_hidden_dims) - 1:
                decoer_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_output))
            else:
                decoer_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_hidden_dims[layer_index + 1]))
                decoer_layers.append(activation)
        self.decoder = nn.Sequential(*decoer_layers)
        #######################################Value#################################################################
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(value_activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(value_activation)
        self.critic = nn.Sequential(*critic_layers)


        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.z_e = None
        self.z_q = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        
        print(f"RMS: {self.rms}")
        print(f"Encoder : {self.encoder}")
        print(f"Codebook: {self.codebook}")
        print(f"Observation Embedding: {self.observation_embd}")
        print(f"Z Embedding: {self.z_embd}")
        print(f"Decoder : {self.decoder}")
        
        
        print(f"Critic: {self.critic}")
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
    def cvqvae_loss(self):
        return self.loss_vq
    
    @property
    def perplexity(self):
        return self.Perplexity

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        observations = self.rms(observations)
        
        self.z_e = self.encoder(observations)
        
        quantized, self.loss_vq, (self.Perplexity, encodings, _) = self.codebook(self.z_e)
        observation_embd = self.observation_embd(observations[:, :self.State_Dimentions])
        #将量化后的向量与观测值进行拼接
        decoder_input = torch.cat((quantized,observation_embd), dim=1)
        
        #计算输出动作
        mean = self.decoder(decoder_input)

        # 使用均值和标准差创建一个正态分布对象
        # 其中标准差为均值乘以0（即不改变均值）再加上self.std
        self.distribution = Normal(mean,self.std)
        #print(f"Distribution: {self.distribution}")
        return mean
    def act(self, observations, **kwargs):
        mean = self.update_distribution(observations)

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

