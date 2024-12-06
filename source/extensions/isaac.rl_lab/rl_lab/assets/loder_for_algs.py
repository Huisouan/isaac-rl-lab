from .base_motionloader import MotionData_Base
import torch
import time
class AmpMotion(MotionData_Base):
    def __init__(self, 
                 data_dir,
                 datatype="isaaclab",
                 file_type="txt",
                 data_spaces = None,
                 env_step_duration = 0.005,**kwargs):
        super().__init__(data_dir,datatype,file_type,data_spaces,env_step_duration,**kwargs)
        self.prepare_amp_state_trans()
    
    def prepare_amp_state_trans(self):
        """
        amp数据预处理，所有数据都被拼接成两个大的二维张量，并且记录每个数据张量的起始和结束行。
        self.amp_state = torch.cat(self.amp_state, dim=0)
        self.amp_state_next = torch.cat(self.amp_state_next, dim=0)
        """
        # 定义数据空间顺序
        amp_data_spaces = [
            "joint_pos",
            "foot_pos",
            "root_lin_vel",
            "root_ang_vel",
            "joint_vel",
            "z_pos",
        ]

        # 构建 rearrange_indices
        rearrange_indices = []
        for item in amp_data_spaces:
            if item in self.cumulative_indices:
                start_index, end_index = self.cumulative_indices[item]
                rearrange_indices.extend(range(start_index, end_index))
            elif item == 'z_pos':
                rearrange_indices.append(42)  # 假设 z_pos 只有一个维度

        # 初始化 amp_state 和 amp_state_next
        self.amp_state = []
        self.amp_state_next = []
        start_end_indices = []
        current_start = 0

        # 处理每个数据张量
        for tragedy in self.data_tensors:
            # 重新排列 tragedy 的列
            rearranged_tragedy = tragedy[:, rearrange_indices]
            s = rearranged_tragedy[:-1]
            s_next = rearranged_tragedy[1:]

            # 记录当前 s 的起始和结束行
            start_end_indices.append((current_start, current_start + s.size(0)))
            current_start += s.size(0)

            # 添加到 amp_state 和 amp_state_next
            self.amp_state.append(s)
            self.amp_state_next.append(s_next)

        # 合并 amp_state 和 amp_state_next 列表为二维张量
        self.amp_state = torch.cat(self.amp_state, dim=0)
        self.amp_state_next = torch.cat(self.amp_state_next, dim=0)

        # 将 start_end_indices 转换为张量
        self.start_end_indices = torch.tensor(start_end_indices, dtype=torch.int64, device=self.device)

        # 计算 max_row_sizes
        self.max_row_sizes = self.start_end_indices[:, 1] - self.start_end_indices[:, 0]
        self.amp_obs_num = self.amp_state.shape[1]
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            
            # 根据权重进行抽样
            indices = torch.multinomial(self.data_weights, mini_batch_size, replacement=True)
            
            # 生成所有需要的 row_idx
            row_indices = torch.rand(mini_batch_size, device=self.device) * self.max_row_sizes[indices]
            row_indices = row_indices.long() 
            
            # 获取每个样本的起始和结束行索引
            start_indices = self.start_end_indices[indices][:, 0]
            
            # 计算实际的行索引
            actual_row_indices = start_indices + row_indices
            
            # 使用切片获取所需的行
            s = self.amp_state[actual_row_indices]
            s_next = self.amp_state_next[actual_row_indices]

            yield s, s_next

class VQVAEMotion(MotionData_Base):
    def __init__(self, 
                 data_dir,
                 datatype="isaaclab",
                 file_type="txt",
                 data_spaces = None,
                 env_step_duration = 0.005,**kwargs):
        super().__init__(data_dir,datatype,file_type,data_spaces,env_step_duration,**kwargs)
        
        def prepare_vqvae_state_trans():
            
            pass