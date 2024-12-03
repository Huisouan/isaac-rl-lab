from .base_motionloader import MotionData_Base
import torch
class AmpMotion(MotionData_Base):
    def __init__(self, 
                 data_dir,
                 datatype="isaaclab",
                 file_type="csv",
                 data_spaces = None,
                 env_step_duration = 0.005,**kwargs):
        super().__init__(data_dir,datatype,file_type,data_spaces,env_step_duration,**kwargs)
        self.prepare_amp_state_trans()
        self.feed_forward_generator(1,5)
    def prepare_amp_state_trans(self):
        self.amp_state = []
        self.amp_state_next = []
        # joint_pos(0-11) foot_pos(12-23) base_lin_vel(24-26) base_ang_vel(27-29) joint_vel(30-41) z_pos(42)
        # 将 amp_data_spaces 转换为有序列表
        amp_data_spaces = [
            "joint_pos",
            "foot_pos",
            "root_lin_vel",
            "root_ang_vel",
            "joint_vel",
            "z_pos" ,
        ]
        rearrange_indices = []
        for items in amp_data_spaces:
            if items in self.cumulative_indices:
                start_index, end_index  =  self.cumulative_indices[items]
                rearrange_indices.extend(range(start_index, end_index))
            elif items =='z_pos':
                start_index, end_index  =(2,3)
                rearrange_indices.extend(range(start_index, end_index))
        for tragedy in self.data_tensors:

            # 重新排列 tragedy 的列
            rearranged_tragedy = tragedy[:, rearrange_indices]
            s = rearranged_tragedy[:-1]
            s_next = rearranged_tragedy[1:]
            self.amp_state.append(s)
            self.amp_state_next.append(s_next)
    
    
        
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            # 根据权重进行抽样
            indices = torch.multinomial(self.data_weights, mini_batch_size, replacement=True)
            
            # 初始化 s 和 s_next 的列表
            s_list = []
            s_next_list = []
            
            for idx in indices:
                # 从 self.amp_state 和 self.amp_state_next 中的每个二维 tensor 随机抽取一行
                row_idx = torch.randint(0, self.amp_state[idx].size(0), (1,))
                s_list.append(self.amp_state[idx][row_idx])
                s_next_list.append(self.amp_state_next[idx][row_idx])
            
            # 将列表转换为 tensor
            s = torch.cat(s_list, dim=0)
            s_next = torch.cat(s_next_list, dim=0)
            
            yield s, s_next