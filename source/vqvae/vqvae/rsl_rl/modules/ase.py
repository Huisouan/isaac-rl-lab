#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import RMSNorm
from ...tasks.utils.wappers.rsl_rl import (
    ASECfg,ASENetcfg,AMPCfg,AMPNetcfg,ASEagentCfg
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
        #build critic 
        if self.seperate_actor_critic == True:
            self.critic_mlp = self._build_mlp(self.mlp_input_num,self.mlp_units)

        #build value
        self.value = self._build_value_layer(input_size=self.mlp_units[-1], output_size=1)
        self.value_activation =  nn.Identity()
    

        
    def _build_disc(self, input_shape):
        # 初始化判别器的MLP
        self._disc_mlp = nn.Sequential()

        mlp_args = {
            'input_size' : input_shape[0], 
            'units' : self._disc_units, 
            'activation' : self._disc_activation, 
            'dense_func' : torch.nn.Linear
        }
        self._disc_mlp = self._build_mlp(**mlp_args)
        
        # 获取MLP输出的大小
        mlp_out_size = self.disc_units[-1]
        # 初始化判别器的对数概率层
        self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

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
        in_size = input        
        for unit in units:
            mlp_layers.append(nn.Linear(in_size, unit))
            mlp_layers.append(self.activation())
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
        
        amp_input_shape = (num_actor_obs, num_critic_obs)   #TODO
        #build network###############################
        
        #build actor and critic net##################
        actor_out_size, critic_out_size = self._build_actor_critic_net(num_actor_obs, self._ase_latent_shape)
        
        #build value net#############################
        self.value = torch.nn.Linear(critic_out_size, self.value_size)  # 价值层
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

            mlp_init = self.initializer  # 编码器初始化器
            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)  # 初始化权重
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)  # 初始化偏置
        else:
            self._enc_mlp = self._disc_mlp  # 使用判别器MLP

        mlp_out_layer = list(self._enc_mlp.modules())[-2]  # 获取MLP的倒数第二层
        mlp_out_size = mlp_out_layer.out_features  # 获取输出特征数
        self._enc = torch.nn.Linear(mlp_out_size, self._ase_latent_shape[-1])  # 编码器线性层
        
        torch.nn.init.uniform_(self._enc.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)  # 初始化权重
        torch.nn.init.zeros_(self._enc.bias)  # 初始化偏置
        
        return
        
    def _build_action_head(self, actor_out_size, num_actions):
        if self.Spacecfg.iscontinuous:
            self.mu = torch.nn.Linear(actor_out_size, num_actions)  # 连续动作的均值层
            self.mu_act = get_activation(self.Spacecfg.mu_activation)  # 均值激活函数none
            mu_init = get_initializer(self.Spacecfg.mu_init,self.mu.weight)  # 均值初始化器
            # 标准差初始化器const_initializer ,nn.init.constant_
            sigma_init = get_initializer(self.Spacecfg.sigma_init,self.sigma,self.Spacecfg.sigma_val)  
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
        style_dim = ase_latent_shape[-1]  # 风格维度

        self.actor_cnn = nn.Sequential()  # 演员CNN
        self.critic_cnn = nn.Sequential()  # 评论家CNN
        
        act_fn = self.activation  # 激活函数是一个relu class
        initializer = self.initializer  # 初始化器

        self.actor_mlp = AMPStyleCatNet1(
            obs_size=input_shape[-1],
            ase_latent_size=ase_latent_shape[-1],
            units=self.mlp_units,
            activation=act_fn,
            style_units=style_units,
            style_dim=style_dim,
            initializer=initializer
        )  # 演员MLP

        if self.separate:
            self.critic_mlp = AMPMLPNet(
                obs_size=input_shape[-1],
                ase_latent_size=ase_latent_shape[-1],
                units=self.mlp_units,
                activation=act_fn,
                initializer=initializer
            )  # 评论家MLP

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

    def forward(self, obs_dict):
        obs = obs_dict['obs']  # 获取观测
        ase_latents = obs_dict['ase_latents']  # 获取ASE潜在变量
        states = obs_dict.get('rnn_states', None)  # 获取RNN状态
        use_hidden_latents = obs_dict.get('use_hidden_latents', False)  # 是否使用隐藏潜在变量

        mu,sigma = self.eval_actor(obs, ase_latents, use_hidden_latents)  # 评估演员
        value = self.eval_critic(obs, ase_latents, use_hidden_latents)  # 评估评论家

        return mu, sigma, value, states


class ASEagent(nn.Module):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        config:ASEagentCfg = ASEagentCfg(),
        num_envs = 1024,
        **kwargs,):
        nn.Module.__init__(self)
        self.a2c_network = ASENet(num_actor_obs,num_critic_obs,num_actions)
        self.aseconf = config
        #init params
        self.num_actor_obs = num_actor_obs
        if self.aseconf.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,)) #   GeneralizedMovingStats((self.value_size,)) #   
        if self.aseconf.normalize_input:
            if isinstance(num_actor_obs, dict):
                self.running_mean_std = RunningMeanStdObs(num_actor_obs)
            else:
                self.running_mean_std = RunningMeanStd(num_actor_obs)       
                
        self._latent_reset_steps = torch.zeros(num_envs, dtype=torch.int32, device='cuda') 
        
    def _update_latents(self):
        # 检查哪些环境需要更新潜在变量
        new_latent_envs = self._latent_reset_steps <= self.vec_env.env.task.progress_buf

        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)  # 重置潜在变量
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                            low=self._latent_steps_min, 
                                                                            high=self._latent_steps_max)
        return
    
            


    def forward(self, input_dict):
        is_train = input_dict.get('is_train', True)
        prev_actions = input_dict.get('prev_actions', None)
        input_dict['obs'] = F.normalize(input_dict['obs'],p=2, dim=1, eps=1e-12)
        #network forward
        mu, logstd, value, states = self.a2c_network(input_dict)
        if self.aseconf.normalize_value:
            value = self.value_mean_std(value)

        
        sigma = torch.exp(logstd)
        result = {}                
        if (is_train):
            amp_obs = input_dict['amp_obs']
            disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
            result["disc_agent_logit"] = disc_agent_logit

            amp_obs_replay = input_dict['amp_obs_replay']
            disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
            result["disc_agent_replay_logit"] = disc_agent_replay_logit

            amp_demo_obs = input_dict['amp_obs_demo']
            disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
            result["disc_demo_logit"] = disc_demo_logit

        if (is_train):
            amp_obs = input_dict['amp_obs']
            enc_pred = self.a2c_network.eval_enc(amp_obs)
            result["enc_pred"] = enc_pred

        return mu,sigma,result        
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        pass

    def reset(self, dones=None):
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

    def update_distribution(self, observations):
        mu,sigma,result = self.forward(observations)
        # 使用均值和标准差创建一个正态分布对象
        # 其中标准差为均值乘以0（即不改变均值）再加上self.std
        self.distribution = Normal(mu, sigma, validate_args=False)
        #print(f"Distribution: {self.distribution}")
        return mu
    
    def act(self, observations, **kwargs):
        mean = self.update_distribution(observations)
        #TODO:这里可以修改贪心算法，
        return self.distribution.sample()
    
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
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
    elif act_name == "none":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None

def get_initializer(initialization, **kwargs):
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
    
    return initializers.get(initialization, lambda v: (print("invalid initializer function"), None))  # 返回默认处理


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