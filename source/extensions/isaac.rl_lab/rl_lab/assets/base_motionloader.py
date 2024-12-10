import os
import csv
import json
import torch
_EPS = torch.finfo(float).eps * 4.0

#from omni.isaac.lab.utils.math import *
#from omni.isaac.lab.utils.math import axis_angle_from_quat ,quat_from_angle_axis,quat_error_magnitude

class MotionData_Base:
    def __init__(self, 
                 data_dir,
                 datatype="isaacgym",
                 file_type="csv",
                 data_spaces = None,
                 env_step_duration = 0.005,
                 
                 **kwargs):
        """
        args:
        data_dir: str 数据目录
        datatype: str 数据类型，即这个数据没转化之前适配什么平台 "isaaclab"和"isaacgym"
        file_type: str  数据文件类型 "csv"和"txt"
        data_spaces: dict 数据空间，即每个数据的维度，用于自定义数据的格式，
                          当没有指定时，会自动根据datatype选择默认的数据格式
        env_step_duration：float 环境步长，即isaaclab每次读数据的间隔
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_dir = data_dir
        
        self.datatype = datatype#TODO 此处定义了数据的类型，如果后面需要从其他的仿真平台导入数据，则根据这个类型来选择不同的数据格式转换函数
        if data_spaces == None:
            self.load_preset_datatype()
        self.calculate_cumulative_indices()  #初始化data的顺序  

        self.env_step_duration = env_step_duration
        self.original_data_tensors = []#存储原始数据的列表
        self.data_tensors = []#存储插值处理后的数据的列表
        self.data_names = []#存储数据名称的列表
        self.data_length = []#总可索引帧数
        self.frame_duration = []#存储帧持续时间的列表
        self.data_time_length = []#存储数据时间长度的列表
        self.data_weights = []#存储数据权重的列表

        # 将 kwargs 中的所有键值对初始化为类的属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if file_type == "csv":
            self.load_csv_data()
        elif file_type == "txt":
            self.load_txt_data()
        else:
            raise ValueError("Invalid file type specified.")
        
        
        #self.data_reordering()
        #从original_data_tensors进行插值，让离散数据的时间间隔与env_step_duration一致，并保存到data_tensors
        self.data_discretization()
        #self.re_calculate_velocity()
        print('Motion Data Loaded Successfully')

    def load_preset_datatype(self):
        if self.datatype == "isaaclab" or self.datatype =="vqvae":
            self.data_spaces = {
                'root_state': 13,
                'joint_pos': 12,
                'joint_vel': 12,
                'foot_pos': 12,
                'foot_vel': 12,
            }
        if self.datatype == "isaacgym" :
            self.data_spaces = {
                'root_pos': 3,
                'root_quat': 4,
                'root_lin_vel': 3,
                'root_ang_vel': 3,
                'joint_pos': 12,
                'joint_vel': 12,
                'foot_pos': 12,
                'foot_vel': 12,
            }

    def data_reordering(self):
        """
        根据data_spaces的值，选择相应的数据转换函数,例如对于isaacgym，会调用self.data_reordering_isaacgym()
        """
        if self.datatype == "isaacgym":
            for i in range(len(self.original_data_tensors)):
                for key, (start, end) in self.cumulative_indices.items():
                    if key == "joint_pos" or key == "joint_vel":
                        # 使用 reorder_from_isaacgym_to_isaacsim_tool 进行转换
                       self.original_data_tensors[i][:,start:end] = self.reorder_from_isaacgym_to_isaacsim_tool( self.original_data_tensors[i][:,start:end])

    def load_txt_data(self):
        """
        从指定目录加载所有txt文件，并将每个文件转换为一个不包含表头的二维Tensor。
        """
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path) as f:
                    motion_json = json.load(f)
                    motion_data = torch.tensor(motion_json["Frames"], device=self.device)
                    # Normalize and standardize quaternions.
                    root_rot = self.root_quat_w(motion_data)
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data = self.write_root_quat(motion_data, root_rot)
                    duration = float(motion_json["FrameDuration"])
                    data_time_length = (motion_data.shape[0] - 1) * duration
                    
                    # 检查 data_time_length 是否为 0
                    if data_time_length <= 0:
                        print(f"Skipping {filename} because data_time_length is 0.")
                        continue  # 跳出当前循环，继续下一个文件      
                                
                    # 将数据加入到数据列表中
                    self.original_data_tensors.append(motion_data)
                    self.data_names.append(filename)
                    self.data_length.append(motion_data.shape[0] - 1)  # 总索引帧数
                    self.frame_duration.append(duration)
                    self.data_time_length.append(data_time_length)
                    self.data_weights.append(float(motion_json["MotionWeight"]))
                    
                    print(f"Loaded {data_time_length}s. motion from {filename}.")

        # 将列表转换为张量
        self.data_length = torch.tensor(self.data_length, device=self.device)
        self.frame_duration = torch.tensor(self.frame_duration, device=self.device)
        self.data_time_length = torch.tensor(self.data_time_length, device=self.device)
        self.data_weights = torch.tensor(self.data_weights, device=self.device)

        # 归一化权重
        total_weight = self.data_weights.sum()
        if total_weight > 0:
            self.data_weights /= total_weight
        else:
            self.data_weights = torch.full_like(self.data_weights, 1.0 / len(self.data_weights))  # 如果总权重为0，均匀分配权重 

    def load_csv_data(self):
        """
        从指定目录加载所有CSV文件，并将每个文件转换为一个不包含表头的二维Tensor。
        """
        self.data_header = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_names.append(filename)
                with open(file_path, newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    header = next(csv_reader)  # 读取表头
                    self.data_header.append(header)  # 存储表头
                    data = []
                    for row in csv_reader:
                        data.append([float(item) for item in row])
                    
                    tensor_data = torch.tensor(data, dtype=torch.float32).to(self.device)
                    self.data_length.append(tensor_data.shape[0])
                    self.original_data_tensors.append(tensor_data)
        self.data_length = torch.tensor(self.data_length, dtype=torch.int64).to(self.device)
        self.data_time_length = torch.tensor(self.data_length * self.frame_duration, dtype=torch.float64).to(self.device)

    def data_discretization(self):
        """
        将数据进行离散化，在MotionData初始化的时候调用，将数据按照仿真环境的step时间间隔进行插值
        以减少强化学习中的计算量
        """
        for original_data_tensor,data_time_length,data_length,name in zip(self.original_data_tensors,self.data_time_length,self.data_length,self.data_names):
            # 创建一个列表来存储插值时间
            interpolation_time_list = []
            interpolation_time = 0
            
            while interpolation_time < data_time_length:
                # 将所有插值的时间点添加到插值时间列表中
                interpolation_time_list.append(interpolation_time)
                interpolation_time += self.env_step_duration
                
            interpolation_time_tensor = torch.tensor(interpolation_time_list,dtype=torch.float64,device=self.device)
            intermidate_data = torch.zeros((len(interpolation_time_tensor),original_data_tensor.shape[1]),device=self.device)
            # 计算 idx_low 和 idx_high
            percentage = interpolation_time_tensor / data_time_length
            #(data_length-1) 是为了确保idx_low和idx_high的范围都在0到data_length-1之间,索引值不能超过data_length-1
            idx_low = torch.floor(percentage *data_length).to(torch.int64)
            idx_high = torch.ceil(percentage * data_length).to(torch.int64)    
            frame_stats = original_data_tensor[idx_low,:]
            frame_ends = original_data_tensor[idx_high,:]
            blend = (percentage * data_length - idx_low).clone().detach().unsqueeze(-1).to(self.device)
           
            for key, (start,end) in self.cumulative_indices.items():
                if key == 'root_quat':
                    interpolatec_quat = self.quaternion_slerp(frame_stats[:,start:end],frame_ends[:,start:end],blend)
                    intermidate_data[:,start:end] = interpolatec_quat
                else:
                    interpolatec_element = self.slerp(frame_stats[:,start:end],frame_ends[:,start:end],blend)
                    intermidate_data[:,start:end] = interpolatec_element
            print(f"Converted : {name}.")
            self.data_tensors.append(intermidate_data)           

    def interpole_frame_at_time(self,interpolation_time,):
        #TODO
        pass

    def calculate_cumulative_indices(self):
        # 计算累计索引
        # 初始化累计索引字典
        self.cumulative_indices = {}
        # 初始化累计索引值
        cumulative_index = 0
        # 遍历数据空间字典
        for key, value in self.data_spaces.items():
            # 计算当前键的累计索引范围，并添加到累计索引字典中
            # key: 字典键
            # cumulative_index: 当前累计索引起始值
            # cumulative_index + value: 当前累计索引结束值
            self.cumulative_indices[key] = (cumulative_index, cumulative_index + value)
            # 更新累计索引值
            cumulative_index += value

    def quaternion_slerp(self,q0: torch.Tensor, q1: torch.Tensor, fraction: torch.Tensor, spin: int = 0, shortestpath: bool = True) -> torch.Tensor:
        """Batch quaternion spherical linear interpolation."""
        
        out = torch.zeros_like(q0)

        # 处理特殊情况
        zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
        ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
        out[zero_mask] = q0[zero_mask]
        out[ones_mask] = q1[ones_mask]

        # 计算点积
        d = torch.sum(q0 * q1, dim=-1, keepdim=True)
        # 限制 d 的取值范围
        d = torch.clamp(d, min=-1.0 + _EPS, max=1.0 - _EPS)

        # 计算 delta
        delta = torch.abs(torch.abs(d) - 1.0)
        dist_mask = (delta < _EPS).squeeze()

        # 处理接近 ±1 的情况
        out[dist_mask] = q0[dist_mask]

        # 计算角度
        angle = torch.acos(d) + spin * torch.pi

        # 处理角度接近0的情况
        angle_mask = (torch.abs(angle) < _EPS).squeeze()
        out[angle_mask] = q0[angle_mask]

        # 选择最短路径
        if shortestpath:
            d_old = torch.clone(d)
            d = torch.where(d_old < 0, -d, d)
            q1 = torch.where(d_old < 0, -q1, q1)

        # 处理剩余情况
        final_mask = torch.logical_or(zero_mask, ones_mask)
        final_mask = torch.logical_or(final_mask, dist_mask)
        final_mask = torch.logical_or(final_mask, angle_mask)
        final_mask = torch.logical_not(final_mask)

        # 计算 1.0 / angle
        isin = 1.0 / angle

        # 计算插值
        q0 *= torch.sin((1.0 - fraction) * angle) * isin
        q1 *= torch.sin(fraction * angle) * isin
        q0 += q1
        out[final_mask] = q0[final_mask]

        return out

    def slerp(self, val0: torch.Tensor, val1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return (1.0 - blend) * val0 + blend * val1 
    
    def reorder_from_isaacsim_to_isaacgym_tool(self, joint_tensor):
        # Convert to a 3x4 tensor
        reshaped_tensor = torch.reshape(joint_tensor, (-1, 3, 4))
        # Transpose the tensor
        transposed_tensor = torch.transpose(reshaped_tensor, 1, 2)
        # Flatten the tensor to 1 dimension
        rearranged_tensor = torch.reshape(transposed_tensor, (-1, 12))
        return rearranged_tensor

    def reorder_from_isaacgym_to_isaacsim_tool(self, rearranged_tensor):
        # Reshape to a 4x3 tensor
        reshaped_tensor = torch.reshape(rearranged_tensor, (-1, 4, 3))
        # Transpose the tensor back
        transposed_tensor = torch.transpose(reshaped_tensor, 1, 2)
        # Flatten the tensor back to the original shape
        original_tensor = torch.reshape(transposed_tensor, (-1, 12))
        return original_tensor

    #############################PROPERTY############################
    def get_frames(self,motion_id,frame_num):
        """
        读取数据tensor，返回一个frame
        """
        return self.data_tensors[motion_id][frame_num]   
     
    def get_tensors(self):
        """
        返回加载的所有数据的列表。
        
        :return: 包含所有CSV文件数据的二维Tensor列表
        """
        return self.data_tensors    
    
    #############READ############################
    #############ROOT############################
    def root_state_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        start, end = self.cumulative_indices['root_state']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    def root_pos_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        start, end = self.cumulative_indices['root_pos']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    def root_quat_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        start, end = self.cumulative_indices['root_quat']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')    

    def root_lin_vel_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        start, end = self.cumulative_indices['root_lin_vel']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')  

    def root_ang_vel_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        start, end = self.cumulative_indices['root_ang_vel']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
    #############ROOT############################
    #############JOINT###########################
    def joint_position_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取关节位置向量。
        
        :param frame: 一维或二维张量
        :return: 关节位置向量
        """
        start, end = self.cumulative_indices['joint_pos']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    def joint_velocity_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取关节速度向量。
        
        :param frame: 一维或二维张量
        :return: 关节速度向量
        """
        start, end = self.cumulative_indices['joint_vel']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
    #############JOINT###########################
    def foot_position_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取脚趾位置向量。
        
        :param frame: 一维或二维张量
        :return: 脚趾位置向量
        """
        start, end = self.cumulative_indices['foot_pos']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    def foot_velocity_w(self, frame: torch.Tensor) -> torch.Tensor:
        """
        提取脚趾速度向量。
        
        :param frame: 一维或二维张量
        :return: 脚趾速度向量
        """
        start, end = self.cumulative_indices['foot_vel']
        if frame.dim() == 1:
            return frame[start:end]
        elif frame.dim() == 2:
            return frame[:, start:end]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    def get_frame_batch_by_timelist_cartpole(self,motion_id, frame_num,state)-> torch.Tensor:
        #此函数仅用于cartpole测试，不适用于其他场景
        frame_parts = []  
        
        for x in self.time_list:  
            frame_step_num = int(x // self.frame_duration)  
            frame = self.get_frame_batch(motion_id, frame_num + frame_step_num)  
            frame = frame -state
            # 拼接 frame_part  
            frame_parts.append(frame)  
        
        # 最后一次性拼接  
        return torch.cat(frame_parts, dim=1)

    ############WRITE############################  

    def write_root_state(self, frame: torch.Tensor, root_state: torch.Tensor) -> torch.Tensor:
        """
        将根状态向量写入帧中。
        
        :param frame: 一维或二维张量
        :param root_state: 根状态向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['root_state']
        if frame.dim() == 1:
            frame[start:end] = root_state
        elif frame.dim() == 2:
            frame[:, start:end] = root_state
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_root_pos(self, frame: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        """
        将根位置向量写入帧中。
        
        :param frame: 一维或二维张量
        :param root_pos: 根位置向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['root_pos']
        if frame.dim() == 1:
            frame[start:end] = root_pos
        elif frame.dim() == 2:
            frame[:, start:end] = root_pos
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_root_quat(self, frame: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        """
        将根四元数向量写入帧中。
        
        :param frame: 一维或二维张量
        :param root_quat: 根四元数向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['root_quat']
        if frame.dim() == 1:
            frame[start:end] = root_quat
        elif frame.dim() == 2:
            frame[:, start:end] = root_quat
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_root_lin_vel(self, frame: torch.Tensor, root_lin_vel: torch.Tensor) -> torch.Tensor:
        """
        将根线速度向量写入帧中。
        
        :param frame: 一维或二维张量
        :param root_lin_vel: 根线速度向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['root_lin_vel']
        if frame.dim() == 1:
            frame[start:end] = root_lin_vel
        elif frame.dim() == 2:
            frame[:, start:end] = root_lin_vel
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_joint_position(self, frame: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        """
        将关节位置向量写入帧中。
        
        :param frame: 一维或二维张量
        :param joint_pos: 关节位置向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['joint_pos']
        if frame.dim() == 1:
            frame[start:end] = joint_pos
        elif frame.dim() == 2:
            frame[:, start:end] = joint_pos
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_joint_velocity(self, frame: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """
        将关节速度向量写入帧中。
        
        :param frame: 一维或二维张量
        :param joint_vel: 关节速度向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['joint_vel']
        if frame.dim() == 1:
            frame[start:end] = joint_vel
        elif frame.dim() == 2:
            frame[:, start:end] = joint_vel
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_foot_position(self, frame: torch.Tensor, foot_pos: torch.Tensor) -> torch.Tensor:
        """
        将脚趾位置向量写入帧中。
        
        :param frame: 一维或二维张量
        :param foot_pos: 脚趾位置向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['foot_pos']
        if frame.dim() == 1:
            frame[start:end] = foot_pos
        elif frame.dim() == 2:
            frame[:, start:end] = foot_pos
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame

    def write_foot_velocity(self, frame: torch.Tensor, foot_vel: torch.Tensor) -> torch.Tensor:
        """
        将脚趾速度向量写入帧中。
        
        :param frame: 一维或二维张量
        :param foot_vel: 脚趾速度向量
        :return: 经过改写的帧
        """
        start, end = self.cumulative_indices['foot_vel']
        if frame.dim() == 1:
            frame[start:end] = foot_vel
        elif frame.dim() == 2:
            frame[:, start:end] = foot_vel
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')
        return frame


@torch.jit.script
def QuaternionNormalize(q: torch.Tensor) -> torch.Tensor:
    """Normalizes the quaternion to length 1.

    Divides the quaternion by its magnitude.  If the magnitude is too
    small, raises a ValueError.

    Args:
    q: A tensor of shape (N, 4) representing N quaternions to be normalized.

    Raises:
    ValueError: If any input quaternion has length near zero.

    Returns:
    A tensor of shape (N, 4) with each quaternion having magnitude 1.
    """
    q_norm = torch.norm(q, dim=1, keepdim=True)
    if torch.any(torch.isclose(q_norm, torch.tensor(0.0))):
        raise ValueError(f"Quaternion may not be zero in QuaternionNormalize: |q| = {q_norm}, q = {q}")
    return q / q_norm

@torch.jit.script
def standardize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    返回一个标准化的四元数，其中 q.w >= 0，以消除 q = -q 的冗余。

    Args:
    q: 要标准化的四元数，形状为 (N, 4)，其中 N 是四元数的数量。

    Returns:
    标准化后的四元数，形状为 (N, 4)。
    """
    # 检查 q 的最后一个维度是否小于 0
    mask = q[:, -1] < 0
    # 对于满足条件的四元数，取其负值
    q[mask] = -q[mask]
    return q

