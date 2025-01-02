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
        self.disc_reward_scale = 2
        self.enc_reward_scale = 1
        #reward weights
        self.task_reward_w  = 0
        self.disc_reward_w  = 0.5
        self.enc_reward_w  = 0.5
        #loss coefs
        self.bounds_loss_coef = 10
        self.disc_logit_reg =  0.01
        self.disc_grad_penalty =  5
        self.disc_coef = 5.0
        self.enc_coef = 5.0
        self.disc_weight_decay:float = 0.0001
        self.amp_diversity_bonus:float = 0.01
        self.latent_steps_min = latent_steps_min
        self.latent_steps_max = latent_steps_max
        
        self.amp_diversity_tar = 1

        self.ase_latent_shape = ase_latent_shape
        self.ase_latents = self.sample_latents(num_envs)
        self.latent_reset_steps = torch.zeros(num_envs, dtype=torch.int32)
        self.reset_latent_step_count(torch.tensor(np.arange(num_envs), dtype=torch.long))
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
        if (len(dones) > 0):
            self.ase_latents[dones] = self.sample_latents(len(dones))  # 重置潜在变量
            self.reset_latent_step_count(dones)  # 重置潜在步数计数   

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

    def update_latents(self,cur_episode_length):
        # 检查哪些环境需要更新潜在变量
        new_latent_envs = self.latent_reset_steps <= cur_episode_length
        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            # 重置潜在变量以及重置潜在步数
            self.ase_latents[new_latent_env_ids] = self.sample_latents(len(new_latent_env_ids))  
            self.reset_latent_step_count(new_latent_env_ids)


    ###########AMP_REWARDS#############################################################    
    def calc_amp_rewards(self, amp_obs):
        # 计算AMP奖励
        with torch.no_grad():
            # 计算判别器的逻辑值
            disc_logits = self.eval_disc(amp_obs)
            # 计算概率值
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            # 计算判别奖励
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001)))
            # 根据配置调整判别奖励
            disc_r *= self.disc_reward_scale

            # 计算编码器奖励
            enc_pred = self.eval_enc(amp_obs)
            err = enc_pred * self.ase_latents
            err = -torch.sum(err, dim=-1, keepdim=True)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self.enc_reward_scale

        return disc_r.squeeze(-1),enc_r.squeeze(-1)

    ############LOSS###################################################################
    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
    def disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # 计算预测损失
        # prediction loss
        #disc_agent_logit是generator产生的数据经过disc出来的结果，disc_demo_logit是数据集数据经过disc出来的结果
        disc_loss_agent = nn.BCEWithLogitsLoss(disc_agent_logit, torch.zeros_like(disc_agent_logit))
        disc_loss_demo = nn.BCEWithLogitsLoss(disc_demo_logit, torch.ones_like(disc_demo_logit))
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # 计算logit正则化损失
        logit_weights = torch.flatten(self.disc_logits.weight)
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self.disc_logit_reg * disc_logit_loss

        # 计算梯度惩罚
        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self.disc_grad_penalty * disc_grad_penalty

        # 计算权重衰减
        # weight decay
        if (self.disc_weight_decay != 0):
            disc_weights = self.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self.disc_weight_decay * disc_weight_decay
            
        return disc_loss

    def enc_loss(self, enc_pred, ase_latent, enc_obs):
        # 计算编码器损失
        #当enc_pred = ase_latent，enc_err最小（负数）
        enc_err = enc_pred * ase_latent
        enc_err = -torch.sum(enc_err, dim=-1, keepdim=True)
        
        enc_loss = torch.mean(enc_err)

        return enc_loss  
    
    def diversity_loss(self, obs, action_params, ase_latents):
        n = obs.shape[0]
        # 获取观测值的数量
        assert(n == action_params.shape[0])
        # 断言行为参数的数量与观测值的数量相等

        new_z = self.sample_latents(n)
        # 从潜在空间中采样新的潜在变量

        mu = self.act(obs=obs, ase_latents=new_z)
        # 计算均值和标准差

        clipped_action_params = torch.clamp(action_params, -1.0, 1.0)
        # 将行为参数限制在[-1.0, 1.0]范围内

        clipped_mu = torch.clamp(mu, -1.0, 1.0)
        # 将均值限制在[-1.0, 1.0]范围内

        a_diff = clipped_action_params - clipped_mu
        # 计算行为参数与均值之间的差异

        a_diff = torch.mean(torch.square(a_diff), dim=-1)
        # 计算差异的平方的均值

        z_diff = new_z * ase_latents
        # 计算新潜在变量与原有潜在变量的点积

        z_diff = torch.sum(z_diff, dim=-1)
        # 计算点积的和

        z_diff = 0.5 - 0.5 * z_diff
        # 对点积的和进行缩放和偏移

        diversity_bonus = a_diff / (z_diff + 1e-5)
        # 计算多样性奖励

        diversity_loss = torch.square(self.amp_diversity_tar - diversity_bonus)
        # 计算多样性损失

        return diversity_loss    
        
    def get_disc_weights(self):
        # 获取判别器所有线性层的权重
        weights = []
        for m in self.disc.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        weights.append(torch.flatten(self.disc_logits.weight))
        return weights
    ############FORWARD################################################################
    def update_distribution(self, observations,ase_latents):
        # Check for NaN values in the observations tensor
        # Compute the mean using the actor network
        style_hid = self.style_net(ase_latents)
        style_embd = self.style_net_out(style_hid)
        observations = torch.cat([observations,style_embd],dim=-1)
        mean = self.actor(observations)
        # Update the distribution
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        if 'ase_latents' in kwargs:
            ase_latents = kwargs['ase_latents']
        else:
            ase_latents = self.ase_latents
        self.update_distribution(observations,ase_latents)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def eval_disc(self, amp_obs):
        disc_mlp_out = self.disc(amp_obs)
        return self.disc_logits(disc_mlp_out)

    def eval_enc(self, amp_obs):
        enc_mlp_out = self.enc(amp_obs)
        return self.enc_logits(enc_mlp_out)

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
