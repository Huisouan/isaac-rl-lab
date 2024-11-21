#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from ..modules import ASEagent
from ..storage import ASERolloutStorage
from ..storage.replay_buffer import ReplayBuffer
from ..datasets_for_txt.motion_loader import AMPLoader
from ..utils.amp_utils import Normalizer

class ASEPPO:
    actor_critic: ASEagent

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        amp_data:AMPLoader = None,
        amp_replay_buffer_size = 100000,
        *args, **kwargs
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate


        # ASE components
        
        # load amp data
        
        self.amp_storage = ReplayBuffer(amp_data.observation_dim, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = Normalizer(amp_data.observation_dim)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = ASERolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Hyper params
        self.normalize_value = True

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        
        latent_shape = self.actor_critic.aseconf.ase_latent_shape
        self.storage = ASERolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, 
            critic_obs_shape, action_shape, latent_shape,self.device
            
        )

    def init_ase_latents(self,num_envs):
        self.actor_critic.init_all_ase_latents(num_envs)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs,amp_obs,cur_episode_length):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.actor_critic._update_latents(cur_episode_length)
        
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.amp_observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos,amp_obs,next_amp_obs_with_term):
        """
        处理环境的步进结果。

        参数:
            rewards: 从环境中获得的奖励。
            dones: 标记表示剧集是否结束。
            infos: 来自环境的额外信息，例如超时。
        """
        # 存储从环境步进中获得的奖励和完成标志
        #计算encoder和disc reward
        amp_obs_trans = torch.cat([amp_obs,next_amp_obs_with_term],dim = -1)
        
        amp_reward = self.actor_critic._calc_amp_rewards(amp_obs_trans)
        #把amp reward加到reward上，此处reward会进入到advantage的计算中，从而影响ppo算法的损失
        self.transition.rewards = self.actor_critic.combine_rewards(rewards.clone(),amp_reward)
        
        self.transition.dones = dones
        
        
        #ase latent
        self.transition.ase_latent = self.actor_critic._ase_latents.detach()
        #将amp obs存入amp replay buffer
        self.amp_storage.insert(self.transition.amp_observations, next_amp_obs_with_term)
        
        # 超时引导
        if "time_outs" in infos:
            # 如果有超时信息，使用它来调整奖励
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        
        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        
        # 重置已完成剧集的演员-评论家网络
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self,):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        amp_rl_trans_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_motion_data_trans_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        
        for ((
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            # ase latent
            ase_latent_batch,
            
            hid_states_batch,
            masks_batch),
            rl_state_trans,
            data_state_trans,
        ) in zip(generator,amp_rl_trans_generator,amp_motion_data_trans_generator):
            self.actor_critic.train_mod = True
            #使用ase_forward
            
            
            self.actor_critic.ase_forward(obs_batch,ase_latent_batch,rl_state_trans,data_state_trans)
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            bound_loss = self.actor_critic.bound_loss(mu_batch)
            
            # 计算判别器损失
            disc_info = self.actor_critic._disc_loss(self.actor_critic.disc_agent_logit, self.actor_critic.disc_demo_logit, data_state_trans)
            disc_loss = disc_info['disc_loss']

            # 计算编码器损失
            enc_latents = self.actor_critic._ase_latents
            enc_info = self.actor_critic._enc_loss(self.actor_critic.enc_pred, enc_latents, rl_state_trans)
            enc_loss = enc_info['enc_loss']


            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                +self.actor_critic.aseconf.bounds_loss_coef * bound_loss +\
                self._disc_coef * disc_loss + self._enc_coef * enc_loss

            if self._enable_amp_diversity_bonus():
                diversity_loss = self.actor_critic._diversity_loss(obs_batch, mu, ase_latent_batch)
                loss += self._amp_diversity_bonus * diversity_loss
                

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
