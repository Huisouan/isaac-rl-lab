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

import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

import learning.amp_datasets as amp_datasets

from tensorboardX import SummaryWriter

class CommonAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, config):
        a2c_common.A2CBase.__init__(self, base_name, config)

        self._load_config_params(config)

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = config.get('clip_actions', True)
        self._save_intermediate = config.get('save_intermediate', False)

        net_config = self._build_net_config()
        #是文件中的Modlsclass的实例，其网络是对应的builder class的实例。
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)

        if self.has_central_value:
            cv_config = {
                'state_shape' : torch_ext.shape_whc_to_cwh(self.state_shape), 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length, 
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len, 
                'model' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'multi_gpu' : self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)
        
        return

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']
        return

    def train(self):
        """
        训练主循环，负责初始化张量、更新模型、记录统计信息等。
        """
        self.init_tensors()  # 初始化张量
        self.last_mean_rewards = -100500  # 初始化最后的平均奖励
        start_time = time.time()  # 记录开始时间
        total_time = 0  # 初始化总时间
        rep_count = 0  # 初始化重复计数
        self.frame = 0  # 初始化帧数
        self.obs = self.env_reset()  # 重置环境并获取初始观测
        self.curr_frames = self.batch_size_envs  # 设置当前帧数为环境批次大小

        model_output_file = os.path.join(self.nn_dir, self.config['name'])  # 设置模型输出文件路径

        if self.multi_gpu:  # 如果使用多 GPU
            self.hvd.setup_algo(self)  # 设置 Horovod

        self._init_train()  # 初始化训练

        while True:  # 主训练循环
            epoch_num = self.update_epoch()  # 更新当前轮次
            train_info = self.train_epoch()  # 训练一个轮次并获取训练信息

            sum_time = train_info['total_time']  # 获取总时间
            total_time += sum_time  # 累加总时间
            frame = self.frame  # 当前帧数
            if self.multi_gpu:  # 如果使用多 GPU
                self.hvd.sync_stats(self)  # 同步统计信息

            if self.rank == 0:  # 如果是主进程
                scaled_time = sum_time  # 缩放时间
                scaled_play_time = train_info['play_time']  # 缩放播放时间
                curr_frames = self.curr_frames  # 当前帧数
                self.frame += curr_frames  # 累加帧数
                if self.print_stats:  # 如果需要打印统计信息
                    fps_step = curr_frames / scaled_play_time  # 计算每秒步骤帧数
                    fps_total = curr_frames / scaled_time  # 计算总帧数
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')  # 打印帧率

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)  # 记录总帧率
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)  # 记录步骤帧率
                self.writer.add_scalar('info/epochs', epoch_num, frame)  # 记录轮次信息
                self._log_train_info(train_info, frame)  # 记录训练信息

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)  # 观察者处理打印后的统计信息

                if self.game_rewards.current_size > 0:  # 如果有游戏奖励
                    mean_rewards = self._get_mean_rewards()  # 获取平均奖励
                    mean_lengths = self.game_lengths.get_mean()  # 获取平均长度

                    for i in range(self.value_size):  # 遍历价值大小
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)  # 记录每帧奖励
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)  # 记录每轮奖励
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)  # 记录每时间奖励

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)  # 记录每帧长度
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)  # 记录每轮长度

                    if self.has_self_play_config:  # 如果有自我对弈配置
                        self.self_play_manager.update(self)  # 更新自我对弈管理器

                if self.save_freq > 0:  # 如果有保存频率
                    if (epoch_num % self.save_freq == 0):  # 每隔一定轮次保存模型
                        self.save(model_output_file)  # 保存模型

                        if (self._save_intermediate):  # 如果需要保存中间模型
                            int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)  # 生成中间模型文件名
                            self.save(int_model_output_file)  # 保存中间模型

                if epoch_num > self.max_epochs:  # 如果超过最大轮次
                    self.save(model_output_file)  # 保存最终模型
                    print('MAX EPOCHS NUM!')  # 打印提示信息
                    return self.last_mean_rewards, epoch_num  # 返回最后的平均奖励和轮次

                update_time = 0  # 初始化更新时间
        return  # 结束训练

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        if (hasattr(self, 'vec_env')):
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

        return

    def train_epoch(self):
        """
        训练一个轮次，包括数据收集、模型更新和统计信息记录。
        """
        play_time_start = time.time()  # 记录数据收集开始时间
        with torch.no_grad():  # 不计算梯度
            if self.is_rnn:  # 如果使用 RNN
                batch_dict = self.play_steps_rnn()  # 收集 RNN 数据
            else:
                batch_dict = self.play_steps()  # 收集普通数据

        play_time_end = time.time()  # 记录数据收集结束时间
        update_time_start = time.time()  # 记录模型更新开始时间
        rnn_masks = batch_dict.get('rnn_masks', None)  # 获取 RNN 掩码

        self.set_train()  # 设置模型为训练模式

        self.curr_frames = batch_dict.pop('played_frames')  # 获取并移除已播放的帧数
        self.prepare_dataset(batch_dict)  # 准备数据集
        self.algo_observer.after_steps()  # 观察者处理步骤后操作

        if self.has_central_value:  # 如果有中心值
            self.train_central_value()  # 训练中心值

        train_info = None  # 初始化训练信息

        if self.is_rnn:  # 如果使用 RNN
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())  # 计算掩码比率
            print(frames_mask_ratio)  # 打印掩码比率

        for _ in range(0, self.mini_epochs_num):  # 进行多个小轮次训练
            ep_kls = []  # 初始化 KL 散度列表
            for i in range(len(self.dataset)):  # 遍历数据集
                curr_train_info = self.train_actor_critic(self.dataset[i])  # 训练 actor-critic 模型

                if self.schedule_type == 'legacy':  # 如果使用旧的学习率调度
                    if self.multi_gpu:  # 如果使用多 GPU
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')  # 平均 KL 散度
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())  # 更新学习率和熵系数
                    self.update_lr(self.last_lr)  # 更新学习率

                if (train_info is None):  # 如果训练信息为空
                    train_info = dict()  # 初始化训练信息字典
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]  # 添加当前训练信息
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)  # 累加当前训练信息

            av_kls = torch_ext.mean_list(train_info['kl'])  # 计算平均 KL 散度

            if self.schedule_type == 'standard':  # 如果使用标准学习率调度
                if self.multi_gpu:  # 如果使用多 GPU
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')  # 平均 KL 散度
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())  # 更新学习率和熵系数
                self.update_lr(self.last_lr)  # 更新学习率

        if self.schedule_type == 'standard_epoch':  # 如果使用标准轮次学习率调度
            if self.multi_gpu:  # 如果使用多 GPU
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')  # 平均 KL 散度
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())  # 更新学习率和熵系数
            self.update_lr(self.last_lr)  # 更新学习率

        update_time_end = time.time()  # 记录模型更新结束时间
        play_time = play_time_end - play_time_start  # 计算数据收集时间
        update_time = update_time_end - update_time_start  # 计算模型更新时间
        total_time = update_time_end - play_time_start  # 计算总时间

        train_info['play_time'] = play_time  # 添加数据收集时间到训练信息
        train_info['update_time'] = update_time  # 添加模型更新时间到训练信息
        train_info['total_time'] = total_time  # 添加总时间到训练信息
        self._record_train_batch_info(batch_dict, train_info)  # 记录训练批次信息

        return train_info  # 返回训练信息

    def play_steps(self):
        """
        收集一个轮次的数据，包括与环境交互、存储经验、计算奖励等。
        """
        self.set_eval()  # 设置模型为评估模式
        
        epinfos = []  # 初始化 episode 信息列表
        done_indices = []  # 初始化完成索引列表
        update_list = self.update_list  # 获取需要更新的数据项列表

        for n in range(self.horizon_length):  # 遍历地平线长度
            self.obs = self.env_reset(done_indices)  # 重置环境并获取新的观测
            self.experience_buffer.update_data('obses', n, self.obs['obs'])  # 更新经验缓冲区中的观测数据

            if self.use_action_masks:  # 如果使用动作掩码
                masks = self.vec_env.get_action_masks()  # 获取动作掩码
                res_dict = self.get_masked_action_values(self.obs, masks)  # 获取带有掩码的动作值
            else:
                res_dict = self.get_action_values(self.obs)  # 获取动作值

            for k in update_list:  # 遍历需要更新的数据项
                self.experience_buffer.update_data(k, n, res_dict[k])  # 更新经验缓冲区中的数据

            if self.has_central_value:  # 如果有中心值
                self.experience_buffer.update_data('states', n, self.obs['states'])  # 更新状态数据

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])  # 与环境交互，获取新的观测、奖励、完成标志和信息
            shaped_rewards = self.rewards_shaper(rewards)  # 调整形奖励
            self.experience_buffer.update_data('rewards', n, shaped_rewards)  # 更新经验缓冲区中的奖励数据
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])  # 更新经验缓冲区中的下一个观测数据
            self.experience_buffer.update_data('dones', n, self.dones)  # 更新经验缓冲区中的完成标志数据

            terminated = infos['terminate'].float()  # 获取终止标志
            terminated = terminated.unsqueeze(-1)  # 增加维度
            next_vals = self._eval_critic(self.obs)  # 评估下一个状态的价值
            next_vals *= (1.0 - terminated)  # 乘以未终止的标志
            self.experience_buffer.update_data('next_values', n, next_vals)  # 更新经验缓冲区中的下一个价值数据

            self.current_rewards += rewards  # 累加当前奖励
            self.current_lengths += 1  # 累加当前长度
            all_done_indices = self.dones.nonzero(as_tuple=False)  # 获取所有完成的索引
            done_indices = all_done_indices[::self.num_agents]  # 获取每个代理的完成索引

            self.game_rewards.update(self.current_rewards[done_indices])  # 更新游戏奖励
            self.game_lengths.update(self.current_lengths[done_indices])  # 更新游戏长度
            self.algo_observer.process_infos(infos, done_indices)  # 处理信息

            not_dones = 1.0 - self.dones.float()  # 获取未完成的标志
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)  # 重置当前奖励
            self.current_lengths = self.current_lengths * not_dones  # 重置当前长度

            done_indices = done_indices[:, 0]  # 获取完成索引的第一列

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()  # 获取完成标志的浮点表示
        mb_values = self.experience_buffer.tensor_dict['values']  # 获取价值数据
        mb_next_values = self.experience_buffer.tensor_dict['next_values']  # 获取下一个价值数据
        mb_rewards = self.experience_buffer.tensor_dict['rewards']  # 获取奖励数据
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)  # 计算优势值
        mb_returns = mb_advs + mb_values  # 计算返回值

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)  # 获取转换后的批次数据
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)  # 添加返回值到批次数据
        batch_dict['played_frames'] = self.batch_size  # 添加已播放的帧数到批次数据

        return batch_dict  # 返回批次数据

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

        return

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss
            
            a_clip_frac = torch.mean(a_info['actor_clipped'].float())
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        return

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
        }
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _init_train(self):
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                    1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()
        
        info = {
            'actor_loss': a_loss,
            'actor_clipped': clipped.detach()
        }
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {
            'critic_loss': c_loss
        }
        return info
    
    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _record_train_batch_info(self, batch_dict, train_info):
        return

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        
        self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)
        return
