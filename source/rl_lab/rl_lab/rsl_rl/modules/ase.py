#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import RMSNorm
from tasks.utils.wappers.rsl_rl import (
    ASECfg,ASENetcfg,AMPCfg,AMPNetcfg
)


DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1
class AMPNet(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        mlp_input_num,
        num_actions,
        Ampcfg :AMPCfg = AMPCfg(),
        Ampnetcfg:AMPNetcfg = AMPNetcfg(),
        **kwargs,
    ):
        # 检查是否有未使用的参数
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        #parameter init##############################
        # 加载判别器的配置
        self.activation = get_activation(Ampnetcfg.activation)
        self.value_act = get_activation(Ampnetcfg.activation)
        
        self.initializer = get_initializer(Ampnetcfg.initializer)
        self.mlp_input_num = mlp_input_num

        self.mlp_units = Ampnetcfg.mlp_units
        self.disc_units = Ampnetcfg.disc_units
        self.enc_units = Ampnetcfg.enc_units
        # 加载判别器的配置
        #############################################
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()        
        self.seperate_actor_critic = Ampnetcfg.separate_disc
        
        #build actor
        self.actor_mlp = self._build_mlp(self.mlp_input_num,self.mlp_units)   
        print('build actor_mlp:', self.actor_mlp)

        #build critic 
        if self.seperate_actor_critic == True:
            self.critic_mlp = self._build_mlp(self.mlp_input_num,self.mlp_units)
            print('build critic_mlp:', self.actor_mlp)
        #build value
        self.value = self._build_value_layer(input_size=self.mlp_units[-1], output_size=1)
        self.value_activation =  nn.Identity()
        print('build value:', self.value)

        
    def _build_disc(self, input_shape):
        # 初始化判别器的MLP
        self._disc_mlp = nn.Sequential()

        mlp_args = {
            'num_input' : input_shape, 
            'units' : self.disc_units, 
        }
        self._disc_mlp = self._build_mlp(**mlp_args)
        print('_build_disc:', self._disc_mlp)
        # 获取MLP输出的大小
        mlp_out_size = self.disc_units[-1]
        # 初始化判别器的对数概率层
        self._disc_logits = torch.nn.Linear(mlp_out_size, 1)
        print('_disc_logits:', self._disc_logits)
        # 初始化MLP的权重
        mlp_init = self.initializer
        for m in self._disc_mlp.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias) 

        # 初始化对数概率层的权重和偏置
        torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self._disc_logits.bias) 

        return   


    def _build_mlp(self,num_input,units):
        input = num_input
        mlp_layers = []
        print('build mlp:', input)
        print('build mlp:', units)
        in_size = input        
        for unit in units:
            mlp_layers.append(nn.Linear(in_size, unit))
            mlp_layers.append(self.activation)
            in_size = unit
        return nn.Sequential(*mlp_layers)

    def _build_value_layer(self,input_size, output_size,):
        return torch.nn.Linear(input_size, output_size)

    def eval_disc(self, amp_obs):
        # 通过MLP处理AMP观测值
        disc_mlp_out = self._disc_mlp(amp_obs)
        # 计算判别器的对数概率
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits
    
    def eval_critic(self, obs):
        # 通过CNN处理观测值
        c_out = self.critic_cnn(obs)
        # 将输出展平
        c_out = c_out.contiguous().view(c_out.size(0), -1)
        # 通过MLP处理展平后的输出
        c_out = self.critic_mlp(c_out)              
        # 计算价值
        value = self.value_act(self.value(c_out))
        return value

    def get_disc_logit_weights(self):
        # 获取判别器对数概率层的权重
        return torch.flatten(self._disc_logits.weight)

    def get_disc_weights(self):
        # 获取判别器所有线性层的权重
        weights = []
        for m in self._disc_mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._disc_logits.weight))
        return weights

class ASENet(AMPNet):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_amp_obs,
        Asecfg :ASECfg = ASECfg(),
        Asenetcfg:ASENetcfg = ASENetcfg(),
        **kwargs,
    ):
        # 检查是否有未使用的参数
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(num_actor_obs,num_actions,)
        #parameter init##############################
        self.initializer = get_initializer(Asenetcfg.initializer)
        self.activation = get_activation(Asenetcfg.activation)
        self._ase_latent_shape =  Asecfg.ase_latent_shape
        
        self.separate = Asenetcfg.separate_disc
        self.mlp_units = Asenetcfg.mlp_units
        self.disc_units = Asenetcfg.disc_units
        self.enc_units = Asenetcfg.enc_units
        self.enc_separate = Asenetcfg.enc_separate
        self.value_size = 1
        self.Spacecfg = Asenetcfg.Spacecfg
        
        amp_input_shape = num_amp_obs   #TODO
        #build network###############################
        
        #build actor and critic net##################
        actor_out_size, critic_out_size = self._build_actor_critic_net(num_actor_obs, self._ase_latent_shape)
        
        #build value net#############################
        self.value = torch.nn.Linear(critic_out_size, self.value_size)  # 价值层
        print("ase value: ", self.value)
        self.value_act = get_activation('none')
        #build action head############################
        self._build_action_head(actor_out_size, num_actions)
        
        mlp_init = self.initializer  # MLP初始化器
        cnn_init = self.initializer  # CNN初始化器
        
        
        #weight init#################################
        for m in self.modules():         
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                cnn_init(m.weight)  # 初始化CNN权重
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)  # 初始化CNN偏置
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)  # 初始化MLP权重
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)  # 初始化MLP偏置       

        self.actor_mlp.init_params()  # 初始化演员MLP参数
        self.critic_mlp.init_params()  # 初始化评论家MLP参数

        #build discriminator and encoder################
        self._build_disc(amp_input_shape)  # 构建判别器
        self._build_enc(amp_input_shape)  # 构建编码器

        return

    def _build_enc(self, input_shape):
        if (self.enc_separate):
            self._enc_mlp = nn.Sequential()  # 编码器MLP
            mlp_args = {
                'input_size': input_shape[0], 
                'units': self.enc_units, 
            }
            self._enc_mlp = self._build_mlp(**mlp_args)  # 构建编码器MLP
            print("ase enc_mlp:",self._enc_mlp)
            mlp_init = self.initializer  # 编码器初始化器
            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)  # 初始化权重
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)  # 初始化偏置
        else:
            self._enc_mlp = self._disc_mlp  # 使用判别器MLP
            print("ase enc_mlp:",self._enc_mlp)
        mlp_out_layer = list(self._enc_mlp.modules())[-1]  # 获取MLP的倒数第二层
        mlp_out_size = mlp_out_layer.out_features  # 获取输出特征数
        self._enc = torch.nn.Linear(mlp_out_size, self._ase_latent_shape)  # 编码器线性层
        print("ase enc:",self._enc)
        
        torch.nn.init.uniform_(self._enc.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)  # 初始化权重
        torch.nn.init.zeros_(self._enc.bias)  # 初始化偏置
        
        return
        
    def _build_action_head(self, actor_out_size, num_actions):
        if self.Spacecfg.iscontinuous:
            self.mu = torch.nn.Linear(actor_out_size, num_actions)  # 连续动作的均值层
            self.mu_act = get_activation(self.Spacecfg.mu_activation)  # 均值激活函数none
            mu_init = get_initializer(self.Spacecfg.mu_init)  # 均值初始化器  # 均值初始化器
            # 标准差初始化器const_initializer ,nn.init.constant_
            sigma_init = get_initializer(self.Spacecfg.sigma_init,val=self.Spacecfg.sigma_val)  
            self.sigma_act = get_activation(self.Spacecfg.sigma_activation)  # 标准差激活函数none
            if (not self.Spacecfg.learn_sigma):
                self.sigma = nn.Parameter(torch.zeros(num_actions, requires_grad=False, dtype=torch.float32), requires_grad=False)  # 固定标准差
            elif  self.Spacecfg.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(num_actions, requires_grad=True, dtype=torch.float32), requires_grad=True)  # 可学习的标准差
            else:
                self.sigma = torch.nn.Linear(actor_out_size, num_actions)  # 动态标准差
            
            #initialize
            mu_init(self.mu.weight)  # 初始化均值层权重
            if self.Spacecfg.fixed_sigma:
                sigma_init(self.sigma)  # 初始化固定标准差
            else:
                sigma_init(self.sigma.weight)  # 初始化动态标准差权重
        
    def _build_actor_critic_net(self, input_shape, ase_latent_shape):
        style_units = [512, 256]  # 风格单元
        style_dim = ase_latent_shape  # 风格维度

        self.actor_cnn = nn.Sequential()  # 演员CNN
        self.critic_cnn = nn.Sequential()  # 评论家CNN
        
        act_fn = self.activation  # 激活函数是一个relu class
        initializer = self.initializer  # 初始化器

        self.actor_mlp = AMPStyleCatNet1(
            obs_size=input_shape,
            ase_latent_size=ase_latent_shape,
            units=self.mlp_units,
            activation=act_fn,
            style_units=style_units,
            style_dim=style_dim,
            initializer=initializer
        )  # 演员MLP
        print("ase actor_mlp:",self.actor_mlp)
        if self.separate:
            self.critic_mlp = AMPMLPNet(
                obs_size=input_shape,
                ase_latent_size=ase_latent_shape,
                units=self.mlp_units,
                activation=act_fn,
                initializer=initializer
            )  # 评论家MLP
        print("ase critic_mlp:",self.critic_mlp)
        actor_out_size = self.actor_mlp.get_out_size()  # 演员输出大小
        critic_out_size = self.critic_mlp.get_out_size()  # 评论家输出大小
        
        return actor_out_size, critic_out_size

    def get_enc_weights(self):
        weights = []
        for m in self._enc_mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))  # 获取编码器MLP的权重

        weights.append(torch.flatten(self._enc.weight))  # 获取编码器权重
        return weights

    def sample_latents(self, n):
        device = next(self._enc.parameters()).device  # 获取设备
        z = torch.normal(torch.zeros([n, self._ase_latent_shape[-1]], device=device))  # 生成正态分布的潜在变量
        z = torch.nn.functional.normalize(z, dim=-1)  # 归一化潜在变量
        return z
                
    def eval_critic(self, obs, ase_latents, use_hidden_latents=False):
        c_out = self.critic_cnn(obs)  # 评论家CNN输出
        c_out = c_out.contiguous().view(c_out.size(0), -1)  # 展平输出
        
        c_out = self.critic_mlp(c_out, ase_latents, use_hidden_latents)  # 评论家MLP输出
        value = self.value_act(self.value(c_out))  # 价值激活
        return value

    def eval_actor(self, obs, ase_latents, use_hidden_latents=False):
        a_out = self.actor_cnn(obs)  # 演员CNN输出
        a_out = a_out.contiguous().view(a_out.size(0), -1)  # 展平输出
        a_out = self.actor_mlp(a_out, ase_latents, use_hidden_latents)  # 演员MLP输出
                    
        mu = self.mu_act(self.mu(a_out))  # 连续动作的均值
        if self.Spacecfg.fixed_sigma:
            sigma = mu * 0.0 + self.sigma_act(self.sigma)  # 固定标准差
        else:
            sigma = self.sigma_act(self.sigma(a_out))  # 动态标准差

        return mu, sigma

    def eval_enc(self, amp_obs):
        enc_mlp_out = self._enc_mlp(amp_obs)  # 编码器MLP输出
        enc_output = self._enc(enc_mlp_out)  # 编码器输出
        enc_output = torch.nn.functional.normalize(enc_output, dim=-1)  # 归一化输出

        return enc_output

    def forward(self, obs,ase_latents,use_hidden_latents = False):
        mu,sigma = self.eval_actor(obs, ase_latents, use_hidden_latents)  # 评估演员
        #value = self.eval_critic(obs, ase_latents, use_hidden_latents)  # 评估评论家
        return mu, sigma

class AMPagent(nn.Module):
    def __init__(
        self,
        mconfig:AMPCfg = AMPCfg(),
        ):
        nn.Module.__init__(self)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ampconf = mconfig
        if self.ampconf.normalize_amp_input:
            self.amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.device)
        
        
    
    def _build_rand_action_probs(self):
        """
        构建随机动作概率数组。

        该方法用于生成一个概率数组，用于决定每个环境中采取随机动作的概率。
        它通过计算环境ID与总环境数的比例来确定每个环境的随机动作概率。
        """
        # 获取环境数量
        num_envs = self.vec_env.env.task.num_envs
        # 将环境ID转换为浮点数张量
        env_ids = torch.tensor(np.arange(num_envs), dtype=torch.float32, device=self.ppo_device)

        # 计算随机动作概率，概率随环境ID递减
        self._rand_action_probs = 1.0 - torch.exp(10 * (env_ids / (num_envs - 1.0) - 1.0))
        # 设置第一个环境的随机动作概率为1.0
        self._rand_action_probs[0] = 1.0
        # 设置最后一个环境的随机动作概率为0.0
        self._rand_action_probs[-1] = 0.0

        # 如果未启用epsilon贪婪策略，则将所有环境的随机动作概率设置为1.0
        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0

        return

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return



    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo


    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return



    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)
    

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = torch.tensor(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    
    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return


    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    

    def update_distribution(self, observations,ase_latents,input_dict):

        pass
    
    def act(self, observations, **kwargs):
        pass
    
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.update_distribution(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):   
        pass 

class ASEagent(AMPagent):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_amp_obs,
        num_envs = 1024,
        config:ASECfg = ASECfg(),
        **kwargs,):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nn.Module.__init__(self)

        self.a2c_network = ASENet(num_actor_obs,num_critic_obs,num_actions,num_amp_obs)
        self.aseconf = config
        #init params
        self.train_mod = True
       
        self.num_actor_obs = num_actor_obs
        if self.aseconf.normalize_value:
            self.value_mean_std = RunningMeanStd(self.a2c_network.value_size) #   GeneralizedMovingStats((self.value_size,)) #   
        if self.aseconf.normalize_input:
            if isinstance(num_actor_obs, dict):
                self.running_mean_std = RunningMeanStdObs(num_actor_obs)
            else:
                self.running_mean_std = RunningMeanStd(num_actor_obs)  
        if self.aseconf.normalize_amp_input:
            self.amp_input_mean_std = RunningMeanStd(num_amp_obs).to(self.device)                 
                     
        self._latent_dim = self.aseconf.ase_latent_shape        
        self._latent_reset_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device) 
        self._ase_latents = torch.zeros((num_envs, self._latent_dim), dtype=torch.float32,
                                         device=self.device)        
        env_ids = torch.tensor(np.arange(num_envs), dtype=torch.long, device=self.device)
        self._reset_latent_step_count(env_ids)
###########LATENTS#################################################################
    def init_all_ase_latents(self,num_envs ):
        # 初始化所有环境的潜在变量
            env_ids = torch.tensor(np.arange(num_envs), dtype=torch.long, device=self.device)
            self._reset_latents(env_ids)  # 重置潜在变量
            self._reset_latent_step_count(env_ids)  # 重置潜在步数计数

    def _update_latents(self,cur_episode_length):
        # 检查哪些环境需要更新潜在变量
        new_latent_envs = self._latent_reset_steps <= cur_episode_length

        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)  # 重置潜在变量
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                            low=self.aseconf.latent_steps_min, 
                                                                            high=self.aseconf.latent_steps_max)
        return
    
    def _reset_latents(self, env_ids):
        # 为指定环境ID重置潜在变量
        if env_ids is None:
            n = len(env_ids)
            z = self._sample_latents(n)
            self._ase_latents[env_ids] = z


    def _sample_latents(self, n):
        # 从模型中采样潜在变量
        z = self.a2c_network.sample_latents(n)
        return z
            
    def _reset_latent_step_count(self, env_ids):
        # 为指定环境ID重置潜在步数计数
        self._latent_reset_steps[env_ids] = torch.randint_like(self._latent_reset_steps[env_ids], low=self.aseconf.latent_steps_min, 
                                                            high=self.aseconf.latent_steps_max)
        return  
###########LATENTS#################################################################

############AMP_OBS################################################################
    def _preproc_amp_obs(self, amp_obs):
        if self.aseconf.normalize_amp_input:
            amp_obs = self.amp_input_mean_std(amp_obs)
        return amp_obs
############AMP_OBS################################################################

###########AMP_REWARDS#############################################################    
    def _calc_amp_rewards(self, amp_obs):
        # 计算AMP奖励
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, self._ase_latents)
        output = {
            'disc_rewards': disc_r,
            'enc_rewards': enc_r
        }
        return output
    
    def combine_rewards(self, task_rewards, amp_rewards):
        # 结合任务奖励和AMP奖励
        disc_r = amp_rewards['disc_rewards']
        enc_r = amp_rewards['enc_rewards']
        combined_rewards = self.aseconf.task_reward_w * task_rewards \
                        + self.aseconf.disc_reward_w * disc_r \
                        + self.aseconf.enc_reward_w * enc_r
        return combined_rewards

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            # 计算判别器的逻辑值
            disc_logits = self._eval_disc(amp_obs)
            # 计算概率值
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            # 计算判别奖励
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            # 根据配置调整判别奖励
            disc_r *= self.aseconf.disc_reward_scale

        return disc_r    
    
    def _calc_enc_rewards(self, amp_obs, ase_latents):
        # 计算编码器奖励
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self.aseconf.enc_reward_scale

        return enc_r 

###########AMP_REWARDS############################################################# 

###########EVALS###################################################################   
    def _eval_actor(self, obs, ase_latents):
        # 评估演员网络
        output = self.a2c_network.eval_actor(obs=obs, ase_latents=ase_latents)
        return output    
    
    def _eval_enc(self, amp_obs):
        # 评估编码器
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.a2c_network.eval_enc(proc_amp_obs)

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.a2c_network.eval_disc(proc_amp_obs)

###########EVALS###################################################################
 
    def _calc_enc_error(self, enc_pred, ase_latent):
        # 计算编码器误差
        # 计算误差值
        # err = enc_pred * ase_latent
        err = enc_pred * ase_latent
        # 对误差值进行求和，并保留维度
        # err = -torch.sum(err, dim=-1, keepdim=True)
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err


    def bound_loss(self, mu):
        if self.aseconf.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
    def _enc_loss(self, enc_pred, ase_latent, enc_obs):
        # 计算编码器损失
        #当enc_pred = ase_latent，enc_err最小（负数）
        enc_err = self._calc_enc_error(enc_pred, ase_latent)
        enc_loss = torch.mean(enc_err)

        # 权重衰减
        if (self.aseconf.enc_weight_decay != 0):
            enc_weights = self.a2c_network.get_enc_weights()
            enc_weights = torch.cat(enc_weights, dim=-1)
            enc_weight_decay = torch.sum(torch.square(enc_weights))
            enc_loss += self.aseconf.enc_weight_decay * enc_weight_decay
            
        enc_info = {
            'enc_loss': enc_loss
        }

        # 如果启用了梯度惩罚，计算梯度惩罚
        if (self._enable_enc_grad_penalty()):
            enc_obs_grad = torch.autograd.grad(enc_err, enc_obs, grad_outputs=torch.ones_like(enc_err),
                                            create_graph=True, retain_graph=True, only_inputs=True)
            enc_obs_grad = enc_obs_grad[0]
            enc_obs_grad = torch.sum(torch.square(enc_obs_grad), dim=-1)
            enc_grad_penalty = torch.mean(enc_obs_grad)

            enc_loss += self.aseconf.enc_grad_penalty * enc_grad_penalty

            enc_info['enc_grad_penalty'] = enc_grad_penalty.detach()

        return enc_info    
    
    def _enable_enc_grad_penalty(self):
        # 检查是否启用了编码器梯度惩罚
        return self.aseconf.enc_grad_penalty != 0    
    
    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # 计算预测损失
        # prediction loss
        #disc_agent_logit是generator产生的数据经过disc出来的结果，disc_demo_logit是数据集数据经过disc出来的结果
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # 计算logit正则化损失
        # logit reg
        logit_weights = self.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # 计算梯度惩罚
        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # 计算权重衰减
        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        # 计算判别器准确率
        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        # 构造返回的信息字典
        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info
    
    def _diversity_loss(self, obs, action_params, ase_latents):
        # 计算多样性损失
        assert(self.a2c_network.is_continuous)
        # 断言a2c网络的输出是连续的

        n = obs.shape[0]
        # 获取观测值的数量
        assert(n == action_params.shape[0])
        # 断言行为参数的数量与观测值的数量相等

        new_z = self._sample_latents(n)
        # 从潜在空间中采样新的潜在变量

        mu, sigma = self._eval_actor(obs=obs, ase_latents=new_z)
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

        diversity_loss = torch.square(self.aseconf.amp_diversity_tar - diversity_bonus)
        # 计算多样性损失

        return diversity_loss    

    
    
    
    @staticmethod
    # not used at the moment
    def forward(self,obs, input_dict):
        return
    def init_weights(sequential, scales):
        pass

    def reset(self, dones=None):
        if (len(dones) > 0):
            self._reset_latents(dones)  # 重置潜在变量
            self._reset_latent_step_count(dones)  # 重置潜在步数计数        
        pass


    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    

    def update_distribution(self, observations,ase_latents):
        observations = F.normalize(observations,p=2, dim=1, eps=1e-12)
        #network forward
        #use_hidden_latents = True时，aselatents会多经过一个隐藏层
        mu, logstd = self.a2c_network(observations,ase_latents,use_hidden_latents = False)
        sigma = torch.exp(logstd)
        # 使用均值和标准差创建一个正态分布对象
        # 其中标准差为均值乘以0（即不改变均值）再加上self.std
        self.distribution = Normal(mu, sigma, validate_args=False)
        #print(f"Distribution: {self.distribution}")
        return mu
    
    def ase_forward(self, observations,ase_latent_batch,rl_state_trans,data_state_trans):
        observations = F.normalize(observations,p=2, dim=1, eps=1e-12)
        #network forward
        #use_hidden_latents = True时，aselatents会多经过一个隐藏层
        mu, logstd = self.a2c_network(observations,ase_latent_batch,use_hidden_latents = False)
        sigma = torch.exp(logstd)
        
        #在这里面，amp_obs_demo就是数据集
        self.disc_agent_logit = self.a2c_network.eval_disc(rl_state_trans)

        self.disc_demo_logit = self.a2c_network.eval_disc(data_state_trans)

        self.enc_pred = self.a2c_network.eval_enc(rl_state_trans)
        # 使用均值和标准差创建一个正态分布对象
        # 其中标准差为均值乘以0（即不改变均值）再加上self.std
        self.distribution = Normal(mu, sigma, validate_args=False)
        #TODO:这里可以修改贪心算法，
        return self.distribution.sample()
    
    def act(self, observations, **kwargs):
        if 'ase_latents' in kwargs:
            ase_latents = kwargs['ase_latents']
        else:
            ase_latents = self._ase_latents
        mean = self.update_distribution(observations,ase_latents)
        #TODO:这里可以修改贪心算法，
        return self.distribution.sample()
    
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.update_distribution(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.a2c_network.eval_critic(critic_observations, self._ase_latents,)
        if self.aseconf.normalize_value:
            value = self.value_mean_std(value)        
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
    elif act_name == "none":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None

def get_initializer(initialization, **kwargs):
    """
    获取神经网络权重的初始化函数。

    Args:
        initialization (str): 初始化方法的名称，支持的选项包括：
            - "xavier_uniform": 使用 Xavier 均匀分布初始化
            - "xavier_normal": 使用 Xavier 正态分布初始化
            - "const_initializer": 使用常量初始化
            - "kaiming_uniform": 使用 Kaiming 均匀分布初始化
            - "kaiming_normal": 使用 Kaiming 正态分布初始化
            - "orthogonal": 使用正交初始化
            - "normal": 使用标准正态分布初始化
            - "default": 默认不进行初始化，直接返回输入
        **kwargs: 其他关键字参数，这些参数将传递给具体的初始化函数

    Returns:
        function: 返回一个初始化函数，该函数接受一个张量作为输入，并对其进行初始化

    Example:
        # 创建一个初始化器
        initializer = get_initializer("xavier_uniform", gain=1.0)
        
        # 获取一个张量
        tensor = torch.empty(3, 5)
        
        # 使用初始化器对张量进行初始化
        initializer(tensor)
    """
    
    initializers = {
        "xavier_uniform": lambda v: nn.init.xavier_uniform_(v, **kwargs),
        "xavier_normal": lambda v: nn.init.xavier_normal_(v, **kwargs),
        "const_initializer": lambda v: nn.init.constant_(v, **kwargs),
        "kaiming_uniform": lambda v: nn.init.kaiming_uniform_(v, **kwargs),
        "kaiming_normal": lambda v: nn.init.kaiming_normal_(v, **kwargs),
        "orthogonal": lambda v: nn.init.orthogonal_(v, **kwargs),
        "normal": lambda v: nn.init.normal_(v, **kwargs),
        "default": lambda v: v  # nn.Identity 不是一个初始化函数，这里直接返回输入
    }
    
    # 返回指定的初始化函数，如果初始化方法无效，则返回一个默认处理函数
    return initializers.get(initialization, lambda v: (print("invalid initializer function"), None))

class AMPMLPNet(torch.nn.Module):
    def __init__(self, obs_size, ase_latent_size, units, activation, initializer):
        super().__init__()  # 调用父类的初始化方法

        input_size = obs_size + ase_latent_size  # 计算输入大小
        print('build amp mlp net:', input_size)  # 打印构建信息
        
        self._units = units  # 存储单元列表
        self._initializer = initializer  # 存储初始化器
        self._mlp = []  # 初始化MLP层列表

        in_size = input_size  # 当前输入大小
        for i in range(len(units)):
            unit = units[i]  # 当前单元大小
            curr_dense = torch.nn.Linear(in_size, unit)  # 创建线性层
            self._mlp.append(curr_dense)  # 添加线性层到列表
            self._mlp.append(activation)  # 添加激活函数到列表
            in_size = unit  # 更新当前输入大小

        self._mlp = nn.Sequential(*self._mlp)  # 将列表转换为Sequential模块
        self.init_params()  # 初始化参数
        return

    def forward(self, obs, latent, skip_style):
        inputs = [obs, latent]  # 输入列表
        input = torch.cat(inputs, dim=-1)  # 拼接输入
        output = self._mlp(input)  # 前向传播
        return output

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):  # 如果是线性层
                self._initializer(m.weight)  # 初始化权重
                if getattr(m, "bias", None) is not None:  # 如果有偏置
                    torch.nn.init.zeros_(m.bias)  # 初始化偏置
        return

    def get_out_size(self):
        out_size = self._units[-1]  # 获取输出大小
        return out_size

class AMPStyleCatNet1(torch.nn.Module):
    def __init__(self, obs_size, ase_latent_size, units, activation,
                 style_units, style_dim, initializer):
        super().__init__()  # 调用父类的初始化方法

        print('build amp style cat net:', obs_size, ase_latent_size)  # 打印构建信息
            
        self._activation = activation  # 存储激活函数RELU
        
        self._initializer = initializer  # 存储初始化器nn.Identity()，不对输入数据进行任何变换，而是直接将输入作为输出返回
        
        self._dense_layers = []  # 是
        self._units = units  # 存储单元列表
        self._style_dim = style_dim  # 存储风格维度
        self._style_activation = torch.tanh  # 存储风格激活函数

        self._style_mlp = self._build_style_mlp(style_units, ase_latent_size)  # 构建风格MLP
        self._style_dense = torch.nn.Linear(style_units[-1], style_dim)  # 构建风格线性层

        in_size = obs_size + style_dim  # 计算输入大小
        for i in range(len(units)):
            unit = units[i]  # 当前单元大小
            out_size = unit  # 输出大小
            curr_dense = torch.nn.Linear(in_size, out_size)  # 创建线性层
            self._dense_layers.append(curr_dense)  # 添加线性层到列表
            
            in_size = out_size  # 更新当前输入大小

        self._dense_layers = nn.ModuleList(self._dense_layers)  # 将列表转换为ModuleList

        self.init_params()  # 初始化参数
        return

    def forward(self, obs, latent, skip_style):
        if (skip_style):
            style = latent  # 如果跳过风格，则直接使用latent
        else:
            style = self.eval_style(latent)  # 否则计算风格

        h = torch.cat([obs, style], dim=-1)  # 拼接观测和风格

        for i in range(len(self._dense_layers)):
            curr_dense = self._dense_layers[i]  # 当前线性层
            h = curr_dense(h)  # 前向传播
            h = self._activation(h)  # 激活

        return h

    def eval_style(self, latent):
        style_h = self._style_mlp(latent)  # 风格MLP输出
        style = self._style_dense(style_h)  # 风格线性层输出
        style = self._style_activation(style)  # 风格激活
        return style

    def init_params(self):
        scale_init_range = 1.0  # 初始化范围

        for m in self.modules():
            if isinstance(m, nn.Linear):  # 如果是线性层
                self._initializer(m.weight)  # 初始化权重
                if getattr(m, "bias", None) is not None:  # 如果有偏置
                    torch.nn.init.zeros_(m.bias)  # 初始化偏置

        nn.init.uniform_(self._style_dense.weight, -scale_init_range, scale_init_range)  # 初始化风格线性层权重
        return

    def get_out_size(self):
        out_size = self._units[-1]  # 获取输出大小
        return out_size

    def _build_style_mlp(self, style_units, input_size):
        in_size = input_size  # 当前输入大小
        layers = []  # 初始化层列表
        for unit in style_units:
            layers.append(torch.nn.Linear(in_size, unit))  # 添加线性层
            layers.append(self._activation)  # 添加激活函数
            in_size = unit  # 更新当前输入大小

        enc_mlp = nn.Sequential(*layers)  # 将列表转换为Sequential模块
        return enc_mlp
   
class RunningMeanStd(nn.Module):
    """_summary_
    Running mean and variance calculation.

    """
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm=False):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)        
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output


        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert(isinstance(insize, dict))
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict({
            k : RunningMeanStd(v, epsilon, per_channel, norm_only) for k,v in insize.items()
        })
    
    def forward(self, input, denorm=False):
        res = {k : self.running_mean_std[k](v, denorm) for k,v in input.items()}
        return res