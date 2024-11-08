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

DISC_LOGIT_INIT_SCALE = 1.0

class AMPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # 调用父类的初始化方法
            nn.Module.__init__(self, **kwargs)
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  


            # 如果是连续动作空间
            if self.is_continuous:
                # 如果不学习sigma
                if (not self.space_config['learn_sigma']):
                    # 获取动作数量
                    actions_num = kwargs.get('actions_num')
                    # 创建sigma初始化器
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    # 创建sigma参数，初始值为0，不需要梯度
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    # 初始化sigma
                    sigma_init(self.sigma)
                    
            # 获取AMP输入形状
            amp_input_shape = kwargs.get('amp_input_shape')
            # 构建判别器
            self._build_disc(amp_input_shape)

            return

        def load(self, params):
            # 调用父类的加载方法
            super().load(params)

            # 加载判别器的配置
            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def forward(self, obs_dict):
            # 获取观测值
            obs = obs_dict['obs']
            # 获取RNN状态
            states = obs_dict.get('rnn_states', None)

            # 评估Actor网络
            actor_outputs = self.eval_actor(obs)
            # 评估Critic网络
            value = self.eval_critic(obs)

            # 返回Actor输出、价值和RNN状态
            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            """
            评估Actor网络，根据观测值生成对应的动作。
            
            Args:
                obs (torch.Tensor): 观测值，形状为 [batch_size, channels, height, width]。
            
            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                    - 如果self.is_discrete为True，则返回动作的对数概率logits，形状为 [batch_size, num_actions]。
                    - 如果self.is_multi_discrete为True，则返回每个维度的动作的对数概率logits的列表，每个logits的形状为 [batch_size, num_actions_dim_i]。
                    - 如果self.is_continuous为True，则返回动作的均值mu和标准差sigma，形状分别为 [batch_size, action_dim]。
            
            """
            # 通过CNN处理观测值
            a_out = self.actor_cnn(obs)
            # 将输出展平
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            # 通过MLP处理展平后的输出
            a_out = self.actor_mlp(a_out)
                        
            # 如果是离散动作空间
            if self.is_discrete:
                # 计算动作的对数概率
                logits = self.logits(a_out)
                return logits

            # 如果是多离散动作空间
            if self.is_multi_discrete:
                # 计算每个维度的动作的对数概率
                logits = [logit(a_out) for logit in self.logits]
                return logits

            # 如果是连续动作空间
            if self.is_continuous:
                # 计算动作的均值
                mu = self.mu_act(self.mu(a_out))
                # 如果sigma固定
                if self.space_config['fixed_sigma']:
                    # 计算固定的sigma
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    # 计算动态的sigma
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

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

        def eval_disc(self, amp_obs):
            # 通过MLP处理AMP观测值
            disc_mlp_out = self._disc_mlp(amp_obs)
            # 计算判别器的对数概率
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

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
            mlp_out_size = self._disc_units[-1]
            # 初始化判别器的对数概率层
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            # 初始化MLP的权重
            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            # 初始化对数概率层的权重和偏置
            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return

    def build(self, name, **kwargs):
        # 构建网络
        net = AMPBuilder.Network(self.params, **kwargs)
        return net
    
