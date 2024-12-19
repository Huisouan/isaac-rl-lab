
import torch
from typing import List, Tuple
from unitree_bridge.config.load_model import load_model
from unitree_bridge.config.algo.algocfg import Algo_def_cfg
from unitree_bridge.config.robot.bot_cfg import Bot_def_cfg

class UnitreeBase():
    def __init__(
        self,
        Algocfg:Algo_def_cfg,
        Botcfg:Bot_def_cfg,
        ):
        #Load model
        self.device  =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = Algocfg.model_path
        self.model = load_model(self.path)
        #Get obs&action dim
        self.obs_dim = Algocfg.policy_observation_dim
        self.action_dim = Algocfg.policy_action_dim
        self.algo_obs = torch.zeros(self.obs_dim, device=self.device)
        self.algo_act = torch.zeros(self.action_dim, device=self.device)
        #Reorder
        self.model_leg_order = Algocfg.joint_order
        self.robot_let_order = Botcfg.joint_order
        self.bot2sim_joint_order = self.compute_index_map(self.robot_let_order, self.model_leg_order)
        self.sim2bot_joint_order = self.compute_index_map(self.model_leg_order, self.robot_let_order)
        
        self.default_jointpos_bias = torch.tensor(Algocfg.default_jointpos_bias,device = self.device)
        
    def compute_index_map(self, src_order: Tuple[str, ...], tgt_order: Tuple[str, ...]) -> torch.Tensor:
        assert len(src_order) == len(tgt_order), f"src_order and tgt_order must have the same length."
        index_map = [src_order.index(joint) for joint in tgt_order]
        return torch.tensor(index_map, dtype=torch.int, device=self.device)
    @staticmethod
    @torch.jit.script
    def joint_reorder(tensor: torch.Tensor, index_map: torch.Tensor) -> torch.Tensor:
        tensor_new = tensor[index_map]
        return tensor_new

    def data_process(self,imu_state,motor_state):

        quaternion = torch.zeros(4)
        for i in range(4):
            quaternion[i] = imu_state.quaternion[i]
        gyroscope = torch.zeros(3)
        for i in range(3):
            gyroscope[i] = imu_state.gyroscope[i]
        joint_pos = torch.zeros(12)
        joint_vel = torch.zeros(12)
        for i in range(12):
            joint_pos[i] = motor_state[i].q
            joint_vel[i] = motor_state[i].dq
        
        joint_pos = joint_pos.to(self.device)
        joint_vel = joint_vel.to(self.device)
        gyroscope = gyroscope.to(self.device)
        quaternion = quaternion.to(self.device)
        # 对关节角度进行重排序
        joint_pos =self.joint_reorder(joint_pos,self.bot2sim_joint_order)
        joint_vel =self.joint_reorder(joint_vel,self.bot2sim_joint_order)

        return gyroscope,quaternion,joint_pos,joint_vel

    def prepare_obs(self,):
        pass
    
    def forward(self):
        pass