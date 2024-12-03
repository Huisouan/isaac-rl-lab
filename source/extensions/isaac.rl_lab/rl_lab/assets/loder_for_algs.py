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
        
    def prepare_amp_state_trans(self):
        self.amp_state_trans = []
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
            state_trans = torch.cat([s, s_next], dim=1)
            
            self.amp_state_trans.append(state_trans)
    
    
        
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        
        pass
    #TODO