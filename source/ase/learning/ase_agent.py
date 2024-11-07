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

from learning import amp_agent 
import numpy as np
import torch.nn as nn
import torch
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from utils import torch_utils
from learning import ase_network_builder

class ASEAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        return

    def init_tensors(self):
        super().init_tensors()
        
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['ase_latents'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self._ase_latents = torch.zeros((batch_shape[-1], self._latent_dim), dtype=torch.float32,
                                         device=self.ppo_device)
        
        self.tensor_list += ['ase_latents']

        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)
        self._reset_latent_step_count(env_ids)

        return
    
    def play_steps(self):
        # 设置评估模式
        self.set_eval()
        
        # 初始化 episode 信息列表和已完成的索引列表
        epinfos = []
        done_indices = []
        update_list = self.update_list

        # 循环执行 horizon_length 步骤
        for n in range(self.horizon_length):
            # 重置环境中的已完成索引
            self.obs = self.env_reset(done_indices)
            # 更新经验缓冲区中的观测值
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            # 更新潜在变量
            self._update_latents()

            # 如果使用动作掩码，则获取带有掩码的动作值
                # 否则，获取普通动作值
            res_dict = self.get_action_values(self.obs, self._ase_latents, self._rand_action_probs)

            # 更新经验缓冲区中的数据
            # self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            # 执行环境步骤并获取新的观测值、奖励、完成标志和信息
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])

            # 对奖励进行整形
            shaped_rewards = self.rewards_shaper(rewards)
            # 更新经验缓冲区中的奖励、下一个观测值、完成标志、AMP 观测值、潜在变量和随机动作掩码
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            self.experience_buffer.update_data('ase_latents', n, self._ase_latents)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            # 获取终止标志并转换为浮点数
            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            # 计算下一个状态的价值
            next_vals = self._eval_critic(self.obs, self._ase_latents)
            # 将未终止的状态的价值乘以 1.0 - 终止标志
            next_vals *= (1.0 - terminated)
            # 更新经验缓冲区中的下一个状态价值
            self.experience_buffer.update_data('next_values', n, next_vals)

            # 累加当前奖励和步长
            self.current_rewards += rewards
            self.current_lengths += 1
            # 获取所有已完成的索引
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            # 更新游戏奖励和长度
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            # 处理信息
            self.algo_observer.process_infos(infos, done_indices)

            # 计算未完成的标志
            not_dones = 1.0 - self.dones.float()

            # 更新当前奖励和步长
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            """
            # 如果有观众，进行 AMP 调试
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos, self._ase_latents)
            """
            
            # 获取已完成的索引
            done_indices = done_indices[:, 0]

        # 获取经验缓冲区中的完成标志、价值、下一个价值和奖励
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        mb_ase_latents = self.experience_buffer.tensor_dict['ase_latents']
        # 计算 AMP 奖励
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_ase_latents)
        # 结合普通奖励和 AMP 奖励
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        
        # 计算优势值
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        # 计算返回值
        mb_returns = mb_advs + mb_values

        # 获取转换后的批次字典
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        # 添加 AMP 奖励到批次字典
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        # 返回批次字典
        return batch_dict

    def get_action_values(self, obs_dict, ase_latents, rand_action_probs):
        # 处理观测值
        processed_obs = self._preproc_obs(obs_dict['obs'])

        # 设置模型为评估模式
        self.model.eval()
        # 准备输入字典
        input_dict = {
            'is_train': False,  # 不是训练模式
            'prev_actions': None,  # 前一个动作
            'obs': processed_obs,  # 处理后的观测值
            'rnn_states': self.rnn_states,  # RNN状态
            'ase_latents': ase_latents  # ASE潜在变量
        }

        # 禁用梯度计算
        with torch.no_grad():
            # 获取模型输出
            res_dict = self.model(input_dict)
            # 如果有中心价值网络，计算中心价值、

        # 如果需要归一化价值，进行归一化
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)

        # 生成随机动作掩码
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0  # 确定性动作掩码
        # 根据随机动作掩码更新动作
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask  # 添加随机动作掩码到输出字典

        return res_dict  # 返回结果字典

    def prepare_dataset(self, batch_dict):
        # 调用父类的准备数据集方法
        super().prepare_dataset(batch_dict)
        
        # 获取ASE潜在变量
        ase_latents = batch_dict['ase_latents']
        self.dataset.values_dict['ase_latents'] = ase_latents  # 将ASE潜在变量添加到数据集中
        
        return

    def calc_gradients(self, input_dict):
        # 设置模型为训练模式
        self.set_train()

        # 获取输入字典中的各种数据
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)  # 处理观测值

        # 处理AMP观测值
        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        """
        if self._enable_enc_grad_penalty():
            amp_obs.requires_grad_(True)  # 启用梯度计算
        """
        # 处理AMP重放缓存观测值
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        # 处理AMP演示观测值
        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)  # 启用梯度计算

        # 获取ASE潜在变量
        ase_latents = input_dict['ase_latents']
        
        # 生成随机动作掩码
        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)  # 随机动作掩码的总和

        # 获取学习率和其他参数
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        # 准备输入字典
        batch_dict = {
            'is_train': True,  # 训练模式
            'prev_actions': actions_batch,  # 前一个动作
            'obs': obs_batch,  # 观测值
            'amp_obs': amp_obs,  # AMP观测值
            'amp_obs_replay': amp_obs_replay,  # AMP重放缓存观测值
            'amp_obs_demo': amp_obs_demo,  # AMP演示观测值
            'ase_latents': ase_latents  # ASE潜在变量
        }

        # 如果模型使用RNN，准备RNN相关数据
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rNN_states']
            batch_dict['seq_length'] = self.seq_len

        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # 获取模型输出
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']
            enc_pred = res_dict['enc_pred']

            # 计算演员损失
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            # 计算评论家损失
            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            # 计算边界损失
            b_loss = self.bound_loss(mu)

            # 计算平均损失
            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum

            # 计算判别器损失
            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']

            # 计算编码器损失
            enc_latents = batch_dict['ase_latents'][0:self._amp_minibatch_size]
            enc_loss_mask = rand_action_mask[0:self._amp_minibatch_size]
            enc_info = self._enc_loss(enc_pred, enc_latents, batch_dict['amp_obs'], enc_loss_mask)
            enc_loss = enc_info['enc_loss']

            # 计算总损失
            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                + self._disc_coef * disc_loss + self._enc_coef * enc_loss

            # 如果启用多样性奖励，计算多样性损失
            if self._enable_amp_diversity_bonus():
                diversity_loss = self._diversity_loss(batch_dict['obs'], mu, batch_dict['ase_latents'])
                diversity_loss = torch.sum(rand_action_mask * diversity_loss) / rand_action_sum
                loss += self._amp_diversity_bonus * diversity_loss
                a_info['amp_diversity_loss'] = diversity_loss

            # 更新演员损失和剪枝比例
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            # 清零梯度
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # 缩放损失并反向传播
        self.scaler.scale(loss).backward()
        # 如果需要截断梯度
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # 计算KL散度
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        # 准备训练结果
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': lr_mul,
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        self.train_result.update(enc_info)

        return
    
    def env_reset(self, env_ids=None):
        # 调用父类的环境重置方法
        obs = super().env_reset(env_ids)
        
        # 如果没有指定环境ID，获取所有环境的ID
        if (env_ids is None):
            num_envs = self.vec_env.env.task.num_envs
            env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)

        # 如果有环境需要重置
        if (len(env_ids) > 0):
            self._reset_latents(env_ids)  # 重置潜在变量
            self._reset_latent_step_count(env_ids)  # 重置潜在步数计数

        return obs  # 返回重置后的观测值

    def _reset_latent_step_count(self, env_ids):
        # 为指定环境ID重置潜在步数计数
        self._latent_reset_steps[env_ids] = torch.randint_like(self._latent_reset_steps[env_ids], low=self._latent_steps_min, 
                                                            high=self._latent_steps_max)
        return

    def _load_config_params(self, config):
        # 调用父类的加载配置参数方法
        super()._load_config_params(config)
        
        # 加载潜在维度和相关参数
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self._latent_dim = config['latent_dim']
        self._amp_diversity_bonus = config['amp_diversity_bonus']
        self._amp_diversity_tar = config['amp_diversity_tar']
        
        # 加载编码器相关参数
        self._enc_coef = config['enc_coef']
        self._enc_weight_decay = config['enc_weight_decay']
        self._enc_reward_scale = config['enc_reward_scale']
        self._enc_grad_penalty = config['enc_grad_penalty']

        self._enc_reward_w = config['enc_reward_w']

        return

    def _build_net_config(self):
        # 调用父类的构建网络配置方法
        config = super()._build_net_config()
        config['ase_latent_shape'] = (self._latent_dim,)  # 添加ASE潜在变量的形状
        return config

    def _reset_latents(self, env_ids):
        # 为指定环境ID重置潜在变量
        n = len(env_ids)
        z = self._sample_latents(n)
        self._ase_latents[env_ids] = z

        # 如果有观众，改变角色颜色
        if (self.vec_env.env.task.viewer):
            self._change_char_color(env_ids)

        return

    def _sample_latents(self, n):
        # 从模型中采样潜在变量
        z = self.model.a2c_network.sample_latents(n)
        return z

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

    def _eval_actor(self, obs, ase_latents):
        # 评估演员网络
        output = self.model.a2c_network.eval_actor(obs=obs, ase_latents=ase_latents)
        return output

    def _eval_critic(self, obs_dict, ase_latents):
        # 评估评论家网络
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs, ase_latents)

        # 如果需要归一化价值，进行归一化
        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        # 计算AMP奖励
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {
            'disc_rewards': disc_r,
            'enc_rewards': enc_r
        }
        return output

    def _calc_enc_rewards(self, amp_obs, ase_latents):
        # 计算编码器奖励
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale

        return enc_r

    def _enc_loss(self, enc_pred, ase_latent, enc_obs, loss_mask):
        # 计算编码器损失
        enc_err = self._calc_enc_error(enc_pred, ase_latent)
        enc_loss = torch.mean(enc_err)

        # 权重衰减
        if (self._enc_weight_decay != 0):
            enc_weights = self.model.a2c_network.get_enc_weights()
            enc_weights = torch.cat(enc_weights, dim=-1)
            enc_weight_decay = torch.sum(torch.square(enc_weights))
            enc_loss += self._enc_weight_decay * enc_weight_decay
            
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

            enc_loss += self._enc_grad_penalty * enc_grad_penalty

            enc_info['enc_grad_penalty'] = enc_grad_penalty.detach()

        return enc_info

    def _diversity_loss(self, obs, action_params, ase_latents):
        # 计算多样性损失
        assert(self.model.a2c_network.is_continuous)
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

        diversity_loss = torch.square(self._amp_diversity_tar - diversity_bonus)
        # 计算多样性损失

        return diversity_loss

    def _calc_enc_error(self, enc_pred, ase_latent):
        # 计算编码器误差
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def _enable_enc_grad_penalty(self):
        # 检查是否启用了编码器梯度惩罚
        return self._enc_grad_penalty != 0

    def _enable_amp_diversity_bonus(self):
        # 检查是否启用了AMP多样性奖励
        return self._amp_diversity_bonus != 0

    def _eval_enc(self, amp_obs):
        # 评估编码器
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs)

    def _combine_rewards(self, task_rewards, amp_rewards):
        # 结合任务奖励和AMP奖励
        disc_r = amp_rewards['disc_rewards']
        enc_r = amp_rewards['enc_rewards']
        combined_rewards = self._task_reward_w * task_rewards \
                        + self._disc_reward_w * disc_r \
                        + self._enc_reward_w * enc_r
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        # 记录训练批次信息
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['enc_rewards'] = batch_dict['enc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        # 记录训练信息
        super()._log_train_info(train_info, frame)
        
        self.writer.add_scalar('losses/enc_loss', torch_ext.mean_list(train_info['enc_loss']).item(), frame)
        
        if (self._enable_amp_diversity_bonus()):
            self.writer.add_scalar('losses/amp_diversity_loss', torch_ext.mean_list(train_info['amp_diversity_loss']).item(), frame)
        
        enc_reward_std, enc_reward_mean = torch.std_mean(train_info['enc_rewards'])
        self.writer.add_scalar('info/enc_reward_mean', enc_reward_mean.item(), frame)
        self.writer.add_scalar('info/enc_reward_std', enc_reward_std.item(), frame)

        if (self._enable_enc_grad_penalty()):
            self.writer.add_scalar('info/enc_grad_penalty', torch_ext.mean_list(train_info['enc_grad_penalty']).item(), frame)

        return


    def _amp_debug(self, info, ase_latents):
        # AMP调试信息
        with torch.no_grad():
            amp_obs = info['amp_obs']
            ase_latents = ase_latents
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs, ase_latents)
            disc_reward = amp_rewards['disc_rewards']
            enc_reward = amp_rewards['enc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            enc_reward = enc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward, enc_reward)
        return