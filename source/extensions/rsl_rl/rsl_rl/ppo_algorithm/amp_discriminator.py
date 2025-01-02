import torch
import torch.nn as nn
from torch import autograd


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.LeakyReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        计算梯度惩罚项。
        
        Args:
            expert_state (torch.Tensor): 专家策略的状态数据。
            expert_next_state (torch.Tensor): 专家策略的下一个状态数据。
            lambda_ (float): 梯度惩罚项的权重，默认为10。
        
        Returns:
            torch.Tensor: 梯度惩罚项的值。
        
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            # 切换模型为评估模式
            self.eval()
            # 如果提供了规范化器，则对状态进行规范化
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            # 将状态和下一个状态拼接，并通过trunk层和amp_linear层计算得到d
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            # 计算奖励
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            # 如果配置了任务奖励的插值比例，则对奖励进行插值
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            # 切换模型为训练模式
            self.train()
        # 返回奖励和d值
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
