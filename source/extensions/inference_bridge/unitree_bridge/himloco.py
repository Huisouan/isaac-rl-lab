import torch

from unitree_bridge.process.base import unitree_base

class him_base(unitree_base):
    def __init__(
        self,
    ):
        super().__init__()
        
    def prepare_obs(self,obs, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions):
        # 确保所有输入张量都是二维的
        base_ang_vel = base_ang_vel.unsqueeze(0) if base_ang_vel.dim() == 1 else base_ang_vel
        projected_gravity = projected_gravity.unsqueeze(0) if projected_gravity.dim() == 1 else projected_gravity
        velocity_commands = velocity_commands.unsqueeze(0) if velocity_commands.dim() == 1 else velocity_commands
        joint_pos = joint_pos.unsqueeze(0) if joint_pos.dim() == 1 else joint_pos
        joint_vel = joint_vel.unsqueeze(0) if joint_vel.dim() == 1 else joint_vel
        actions = actions.unsqueeze(0) if actions.dim() == 1 else actions
        
        # 拼接所有输入张量
        single_obs = torch.cat([base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions], dim=-1)
        
        # 确保 obs 是二维的
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 拼接 single_obs 到 obs 的前面
        obs = torch.cat([single_obs, obs[:, :-45]], dim=1).to(torch.float32)
        
        return obs        
    
    def forward(self,imu_data,motor_state):
        gyroscope,quaternion,joint_pos,joint_vel = 
