from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories

from .rollout_storage import RolloutStorage

class ASERolloutStorage(RolloutStorage):
    #从RolloutStorage中继承的新类，增加了ASE latent的数据
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.ase_latent = None
    
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, latent_shape = 64,device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)
        #添加一个新的属性ase_latent，用于存储ASE latent
        self.ase_latent = torch.zeros(num_transitions_per_env, num_envs,latent_shape, device=self.device)

    def add_transitions(self, transition: Transition):
        # 如果当前步骤数已经超过了每个环境的过渡数，则抛出异常
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        # 将transition中的观测值复制到self.observations对应步骤的位置
        self.observations[self.step].copy_(transition.observations)

        # 如果self.privileged_observations不为空，则将transition中的critic观测值复制到self.privileged_observations对应步骤的位置
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)

        # 将transition中的动作复制到self.actions对应步骤的位置
        self.actions[self.step].copy_(transition.actions)

        # 将transition中的奖励复制到self.rewards对应步骤的位置，并调整维度
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))

        # 将transition中的结束状态复制到self.dones对应步骤的位置，并调整维度
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # 将transition中的值复制到self.values对应步骤的位置
        self.values[self.step].copy_(transition.values)

        # 将transition中的动作对数概率复制到self.actions_log_prob对应步骤的位置，并调整维度
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))

        # 将transition中的动作均值复制到self.mu对应步骤的位置
        self.mu[self.step].copy_(transition.action_mean)

        # 将transition中的动作标准差复制到self.sigma对应步骤的位置
        self.sigma[self.step].copy_(transition.action_sigma)


        self.ase_latent[self.step].copy_(transition.ase_latent)

        # 保存transition中的隐藏状态
        self._save_hidden_states(transition.hidden_states)

        # 增加步骤数
        self.step += 1
        
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # 计算每个小批量的大小
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # 生成随机索引
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # 展开观测值
        observations = self.observations.flatten(0, 1)

        # 如果存在特权观测值，则使用特权观测值，否则使用普通观测值
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            # 使用普通观测值作为critic观测值
            critic_observations = observations

        # 展开动作、值、回报、旧动作对数概率、优势、旧均值和旧标准差
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # 训练多轮
        for epoch in range(num_epochs):
            # 分割数据为小批量
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # 提取当前小批量的数据
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                ase_latent_batch = self.ase_latent[batch_idx]

                # 产出当前小批量的数据，以及一些额外的占位符
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,ase_latent_batch, (
                    None,
                    None,
                ), None