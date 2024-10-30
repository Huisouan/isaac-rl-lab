# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np
import enum

from learning import amp_network_builder

ENC_LOGIT_INIT_SCALE = 0.1

class LatentType(enum.Enum):
    uniform = 0
    sphere = 1

class ASEBuilder(amp_network_builder.AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法
        return

    class Network(amp_network_builder.AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get('actions_num')  # 获取动作数量
            input_shape = kwargs.get('input_shape')  # 获取输入形状
            self.value_size = kwargs.get('value_size', 1)  # 获取价值大小，默认为1
            self.num_seqs = num_seqs = kwargs.get('num_seqs', 1)  # 获取序列数量，默认为1
            amp_input_shape = kwargs.get('amp_input_shape')  # 获取AMP输入形状
            self._ase_latent_shape = kwargs.get('ase_latent_shape')  # 获取ASE潜在形状

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)  # 初始化基类
            
            self.load(params)  # 加载参数

            actor_out_size, critic_out_size = self._build_actor_critic_net(input_shape, self._ase_latent_shape)  # 构建演员-评论家网络

            self.value = torch.nn.Linear(critic_out_size, self.value_size)  # 价值层
            self.value_act = self.activations_factory.create(self.value_activation)  # 价值激活函数
            
            if self.is_discrete:
                self.logits = torch.nn.Linear(actor_out_size, actions_num)  # 离散动作的对数概率层
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num])  # 多离散动作的对数概率层
            if self.is_continuous:
                self.mu = torch.nn.Linear(actor_out_size, actions_num)  # 连续动作的均值层
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])  # 均值激活函数
                mu_init = self.init_factory.create(**self.space_config['mu_init'])  # 均值初始化器
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])  # 标准差激活函数

                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])  # 标准差初始化器

                if (not self.space_config['learn_sigma']):
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)  # 固定标准差
                elif self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)  # 可学习的标准差
                else:
                    self.sigma = torch.nn.Linear(actor_out_size, actions_num)  # 动态标准差

            mlp_init = self.init_factory.create(**self.initializer)  # MLP初始化器
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])  # CNN初始化器

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

            if self.is_continuous:
                mu_init(self.mu.weight)  # 初始化均值层权重
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)  # 初始化固定标准差
                else:
                    sigma_init(self.sigma.weight)  # 初始化动态标准差权重

            self._build_disc(amp_input_shape)  # 构建判别器
            self._build_enc(amp_input_shape)  # 构建编码器

            return
        
        def load(self, params):
            super().load(params)  # 调用父类的加载方法

            self._enc_units = params['enc']['units']  # 编码器单元
            self._enc_activation = params['enc']['activation']  # 编码器激活函数
            self._enc_initializer = params['enc']['initializer']  # 编码器初始化器
            self._enc_separate = params['enc']['separate']  # 编码器是否分离

            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']  # 获取观测
            ase_latents = obs_dict['ase_latents']  # 获取ASE潜在变量
            states = obs_dict.get('rnn_states', None)  # 获取RNN状态
            use_hidden_latents = obs_dict.get('use_hidden_latents', False)  # 是否使用隐藏潜在变量

            actor_outputs = self.eval_actor(obs, ase_latents, use_hidden_latents)  # 评估演员
            value = self.eval_critic(obs, ase_latents, use_hidden_latents)  # 评估评论家

            output = actor_outputs + (value, states)  # 组合输出

            return output

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
                     
            if self.is_discrete:
                logits = self.logits(a_out)  # 离散动作的对数概率
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]  # 多离散动作的对数概率
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))  # 连续动作的均值
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)  # 固定标准差
                else:
                    sigma = self.sigma_act(self.sigma(a_out))  # 动态标准差

                return mu, sigma
            return

        def get_enc_weights(self):
            weights = []
            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))  # 获取编码器MLP的权重

            weights.append(torch.flatten(self._enc.weight))  # 获取编码器权重
            return weights

        def _build_actor_critic_net(self, input_shape, ase_latent_shape):
            style_units = [512, 256]  # 风格单元
            style_dim = ase_latent_shape[-1]  # 风格维度

            self.actor_cnn = nn.Sequential()  # 演员CNN
            self.critic_cnn = nn.Sequential()  # 评论家CNN
            
            act_fn = self.activations_factory.create(self.activation)  # 激活函数
            initializer = self.init_factory.create(**self.initializer)  # 初始化器

            self.actor_mlp = AMPStyleCatNet1(
                obs_size=input_shape[-1],
                ase_latent_size=ase_latent_shape[-1],
                units=self.units,
                activation=act_fn,
                style_units=style_units,
                style_dim=style_dim,
                initializer=initializer
            )  # 演员MLP

            if self.separate:
                self.critic_mlp = AMPMLPNet(
                    obs_size=input_shape[-1],
                    ase_latent_size=ase_latent_shape[-1],
                    units=self.units,
                    activation=act_fn,
                    initializer=initializer
                )  # 评论家MLP

            actor_out_size = self.actor_mlp.get_out_size()  # 演员输出大小
            critic_out_size = self.critic_mlp.get_out_size()  # 评论家输出大小

            return actor_out_size, critic_out_size

        def _build_enc(self, input_shape):
            if (self._enc_separate):
                self._enc_mlp = nn.Sequential()  # 编码器MLP
                mlp_args = {
                    'input_size': input_shape[0], 
                    'units': self._enc_units, 
                    'activation': self._enc_activation, 
                    'dense_func': torch.nn.Linear
                }
                self._enc_mlp = self._build_mlp(**mlp_args)  # 构建编码器MLP

                mlp_init = self.init_factory.create(**self._enc_initializer)  # 编码器初始化器
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

        def eval_enc(self, amp_obs):
            enc_mlp_out = self._enc_mlp(amp_obs)  # 编码器MLP输出
            enc_output = self._enc(enc_mlp_out)  # 编码器输出
            enc_output = torch.nn.functional.normalize(enc_output, dim=-1)  # 归一化输出

            return enc_output

        def sample_latents(self, n):
            device = next(self._enc.parameters()).device  # 获取设备
            z = torch.normal(torch.zeros([n, self._ase_latent_shape[-1]], device=device))  # 生成正态分布的潜在变量
            z = torch.nn.functional.normalize(z, dim=-1)  # 归一化潜在变量
            return z

    def build(self, name, **kwargs):
        net = ASEBuilder.Network(self.params, **kwargs)  # 构建网络
        return net


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
            
        self._activation = activation  # 存储激活函数
        self._initializer = initializer  # 存储初始化器
        self._dense_layers = []  # 初始化密集层列表
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