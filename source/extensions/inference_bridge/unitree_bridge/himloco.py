import torch

from unitree_bridge.process.base import UnitreeBase

from unitree_bridge.config.algo.algocfg import HIMConfig
from unitree_bridge.config.robot.bot_cfg import GO2

from unitree_bridge.config.env import quat_rotate_inverse,GRAVITY_VEC

class Himloco(UnitreeBase):
    def __init__(self,Algocfg:HIMConfig,Botcfg:GO2,):
        super().__init__(Algocfg,Botcfg)
        self.base_ang_vel_scale = Algocfg.base_ang_vel_scale
        self.joint_pos_scale = Algocfg.joint_pos_scale
        self.joint_vel_scale = Algocfg.joint_vel_scale
        self.actions_joint_pos_scale = Algocfg.actions_joint_pos_scale
        
    def prepare_obs(self,base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel):
        """函数输入从机器人数据中预处理得到的数据
            对数据进行缩放操作
            并将数据整合到self.obs中
        """
        base_ang_vel = base_ang_vel.unsqueeze(0) if base_ang_vel.dim() == 1 else base_ang_vel
        projected_gravity = projected_gravity.unsqueeze(0) if projected_gravity.dim() == 1 else projected_gravity
        velocity_commands = velocity_commands.unsqueeze(0) if velocity_commands.dim() == 1 else velocity_commands
        joint_pos = joint_pos.unsqueeze(0) if joint_pos.dim() == 1 else joint_pos
        joint_vel = joint_vel.unsqueeze(0) if joint_vel.dim() == 1 else joint_vel
        
        
        # 拼接所有输入张量
        single_obs = torch.cat([base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, self.actions], dim=-1)
        
        # 拼接 single_obs 到 obs 的前面
        self.obs = torch.cat([single_obs, self.obs[:, :-45]], dim=1).to(torch.float32)
        
    
    def forward(self,imu_data,motor_state,velocity_commands):
        with torch.inference_mode():
            # 数据预处理
            gyroscope,quaternion,joint_pos,joint_vel = self.data_process(imu_data,motor_state)
            projected_gravity = quat_rotate_inverse(quaternion,GRAVITY_VEC)
            #观测值缩放
            self.prepare_obs(gyroscope, projected_gravity, velocity_commands, joint_pos, joint_vel)
            #模型推理
            self.algo_act = self.model(self.obs)
            #输出缩放
            bot_act = self.algo_act*self.actions_joint_pos_scale + self.default_jointpos_bias
            bot_act = self.joint_reorder(bot_act,self.sim2bot_joint_order)
            bot_act = bot_act.cpu().detach().numpy()
            return bot_act