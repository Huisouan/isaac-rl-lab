
import torch
from unitree_bridge.config.load_model import load_model
from unitree_bridge.config.algo.algocfg import Algo_def_cfg
from unitree_bridge.config.robot.bot_cfg import Bot_def_cfg

class unitree_base():
    def __init__(
        self,
        Algocfg:Algo_def_cfg,
        Botcfg:Bot_def_cfg,
        ):
        self.device  =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = Algocfg.model_path
        self.model = load_model(self.path)
        
        self.model_leg_order = Algocfg.joint_order
        self.robot_let_order = Botcfg.joint_order
        self.default_jointpos_bias = torch.tensor(Algocfg.default_jointpos_bias,device = self.device)
        
    def joint_reorder(self,tensor, src_order, tgt_order):
        assert len(src_order) == len(tgt_order), f"src_order and tgt_order must have same length."
        index_map = [src_order.index(joint) for joint in tgt_order]
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
        # 对关节角度进行重排序
        joint_pos =self.joint_reorder(joint_pos,self.robot_let_order,self.model_leg_order).to(self.device)
        joint_vel =self.joint_reorder(joint_vel,self.robot_let_order,self.model_leg_order).to(self.device)
        #将joint_pos减去默认关节位置偏移量
        joint_pos = joint_pos - self.default_jointpos_bias
        
        return gyroscope,quaternion,joint_pos,joint_vel



    def prepare_obs(self,):
        pass
    
    def forward(self):
        pass