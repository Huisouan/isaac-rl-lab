import torch
import pandas as pd
import os

from unitree_bridge.process.base import UnitreeBase
from unitree_bridge.config.algo.algocfg import HIMConfig
from unitree_bridge.config.robot.bot_cfg import GO2
from unitree_bridge.config.env import quat_rotate_inverse, GRAVITY_VEC

class Himloco(UnitreeBase):
    def __init__(self, Algocfg: HIMConfig, Botcfg: GO2):
        super().__init__(Algocfg, Botcfg)
        self.base_ang_vel_scale = Algocfg.base_ang_vel_scale
        self.joint_pos_scale = Algocfg.joint_pos_scale
        self.joint_vel_scale = Algocfg.joint_vel_scale
        self.actions_joint_pos_scale = Algocfg.actions_joint_pos_scale
        self.reset_action = False
        # 创建目录用于保存数据
        self.data_dir = "recorddata"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 记录文件路径
        self.record_file_path = os.path.join(self.data_dir, "algo_obs.csv")
        self.first_write = True  # 用于判断是否第一次写入数据

    def prepare_obs(self, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel):
        """函数输入从机器人数据中预处理得到的数据
            对数据进行缩放操作
            并将数据整合到self.obs中
        """ 
        base_ang_vel *= self.base_ang_vel_scale
        # 将joint_pos减去默认关节位置偏移量
        joint_pos = (joint_pos - self.default_jointpos_bias)*self.joint_pos_scale
        
        joint_vel *= self.joint_vel_scale
        
        # 拼接所有输入张量
        single_obs = torch.cat([base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, self.algo_act], dim=-1)
        
        # 拼接 single_obs 到 obs 的前面
        self.algo_obs = torch.cat((single_obs, self.algo_obs[:-45]), dim=-1)
        
        # 记录 algo_obs 数据
        self.record_algo_obs(single_obs)
    def forward(self, imu_data, motor_state, velocity_commands,Kp,Kd):
        with torch.inference_mode():
            # 数据预处理
            gyroscope, quaternion, joint_pos, joint_vel = self.data_process(imu_data, motor_state)
            projected_gravity = quat_rotate_inverse(quaternion, GRAVITY_VEC)
            # 观测值缩放
            velocity_command = torch.tensor(velocity_commands, device=self.device)
            
            self.prepare_obs(gyroscope, projected_gravity, velocity_command, joint_pos, joint_vel)
            # 模型推理
            self.algo_act = self.model(self.algo_obs.unsqueeze(0)).squeeze(0)
            if self.reset_action:
                self.algo_act = torch.zeros_like(self.algo_act)            
            # 输出缩放
            bot_act = self.algo_act * self.actions_joint_pos_scale + self.default_jointpos_bias

            action_up_limit = (45+Kd*joint_vel) / Kp
            action_down_limit = (-45+Kd*joint_vel) / Kp

            bot_act = torch.clamp(bot_act, action_down_limit, action_up_limit)

            bot_act = self.joint_reorder(bot_act, self.sim2bot_joint_order)

            bot_act = bot_act.cpu().detach().numpy()
            return bot_act
    def record_algo_obs(self, single_obs):
        """记录 algo_obs 数据到 CSV 文件"""
        # 根据观察值的具体含义定义表头
        observation_names = [
            "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "velocity_command_x", "velocity_command_y", "velocity_command_z",
            "joint_pos_0", "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5",
            "joint_pos_6", "joint_pos_7", "joint_pos_8", "joint_pos_9", "joint_pos_10", "joint_pos_11",
            "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5",
            "joint_vel_6", "joint_vel_7", "joint_vel_8", "joint_vel_9", "joint_vel_10", "joint_vel_11",
            "action_0", "action_1", "action_2", "action_3", "action_4", "action_5",
            "action_6", "action_7", "action_8", "action_9", "action_10", "action_11"
        ]
        
        data_dict = {name: [value] for name, value in zip(observation_names, single_obs.tolist())}
        
        df = pd.DataFrame(data_dict)
        
        if self.first_write:
            df.to_csv(self.record_file_path, mode='w', index=False)
            self.first_write = False
        else:
            df.to_csv(self.record_file_path, mode='a', header=False, index=False)

