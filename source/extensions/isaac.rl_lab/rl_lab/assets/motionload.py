import os
import csv
import json
import torch
_EPS = torch.finfo(float).eps * 4.0

#from omni.isaac.lab.utils.math import *
#from omni.isaac.lab.utils.math import axis_angle_from_quat ,quat_from_angle_axis,quat_error_magnitude
class MotionData:
    def __init__(self, 
                 data_dir,datatype="isaaclab",
                 file_type="csv",
                 data_spaces = None,
                 env_step_duration = 0.005,**kwargs):
        """
        初始化方法，设置数据目录。
        
        :param data_dir: 包含CSV文件的目录路径
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
        self.data_length = []#存储数据长度的列表
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
        if self.datatype == "amp" or self.datatype == "ase":
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

        if self.datatype == "vqvae":
            pass
        
    def load_txt_data(self):
        """
        从指定目录加载所有txt文件，并将每个文件转换为一个不包含表头的二维Tensor。
        """
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
            with open(file_path) as f:
                motion_json = json.load(f)
                motion_data = torch.tensor(motion_json["Frames"])
                # Normalize and standardize quaternions.
                root_rot = self.root_quat_w(motion_data)
                root_rot = QuaternionNormalize(root_rot)
                root_rot = standardize_quaternion(root_rot)
                motion_data = self.write_root_quat(motion_data, root_rot)
                duration = float(motion_json["FrameDuration"])
                data_time_length = (motion_data.shape[0] - 1) * duration
                
                #将数据加入到数据列表中
                self.original_data_tensors.append(motion_data)
                self.data_names.append(filename)
                self.data_length.append(motion_data.shape[0])#总帧数
                self.frame_duration.append(duration)
                self.data_time_length.append(data_time_length)
                self.data_weights.append(float(motion_json["MotionWeight"]))
                
                print(f"Loaded {data_time_length}s. motion from {filename}.")

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
        for original_data_tensor,data_time_length,data_length in zip(self.original_data_tensors,self.data_time_length,self.data_length):
            interpolation_time = 0
            while interpolation_time < data_time_length:
                # 计算 idx_low 和 idx_high
                percentage = interpolation_time / data_time_length
                idx_low = torch.floor(percentage * data_length).to(torch.int64)
                idx_high = torch.ceil(percentage * data_length).to(torch.int64)    
                frame_stats = original_data_tensor[idx_low]
                frame_ends = original_data_tensor[idx_high]
                
                for key, (start,end) in self.cumulative_indices.items():
                    if key == 'root_quat':
                #TODO
                        
                
                
    def interpole_frame_at_time(self,interpolation_time,):
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

    def re_calculate_velocity(self):
        # 矫正数据集中的速度
        # Root state [pos, quat, lin_vel, ang_vel] in simulation world frame. Shape is (num_instances, 13)
        self.data_tensors = []  # 清空已校正的数据列表

        for original_data_tensors in self.original_data_tensors:
            # 插值计算基本线性速度
            root_state = self.root_state_w(original_data_tensors)
            joint_pos = self.joint_position_w(original_data_tensors)
            
            # 获取当前帧和下一帧的数据
            num_instances = root_state.shape[0]
            root_state_next = torch.roll(root_state, shifts=-1, dims=0)
            
            # 计算线速度
            base_lin_vel = self.base_lin_vel_interpolation(root_state[:, 0:3], root_state_next[:, 0:3], self.frame_duration)
            
            # 计算角速度
            base_ang_vel = self.base_ang_vel_interpolation(root_state[:, 3:7], root_state_next[:, 3:7], self.frame_duration)
            
            # 插值计算关节速度
            joint_pos_next = torch.roll(joint_pos, shifts=-1, dims=0)
            joint_vel = (joint_pos_next - joint_pos) / self.frame_duration
            
            # 设置最后一帧的速度为0,我们在实际使用数据集的时候并不会用到这一帧，这里只是为了让tensor的形状保持相同
            base_lin_vel[-1] = 0.0
            base_ang_vel[-1] = 0.0
            joint_vel[-1] = 0.0
            
            # 创建一个新的张量副本
            new_tensor = original_data_tensors.clone()
            
            # 更新根状态的速度信息
            new_tensor[:, 7:10] = base_lin_vel
            new_tensor[:, 10:13] = base_ang_vel
            
            # 更新关节速度信息
            new_tensor = self.write_joint_velocity(new_tensor, joint_vel)
            
            # 将校正后的张量添加到列表中
            self.data_tensors.append(new_tensor)

    def get_random_frame_batch(self,batch_size):
        """
        从数据集中获取随即的一批数据，记录下随机的帧数和对应的motion_id以及数据的长度。
        用于环境的初始化中。
        """
        random_frame_id = torch.randint(0,len(self.original_data_tensors),size=(batch_size,)).to(self.device)
        random_frame_index = torch.rand((1,len(random_frame_id)))[0].to(self.device)
        datalength = self.data_length[random_frame_id]-self.motion_bias
        rand_frames = torch.floor(datalength*random_frame_index)
        
        
        #####################################################################################
        # 保存随机帧的索引的第二方法
        self.random_frame_id = random_frame_id
        self.rand_frames = rand_frames
        self.datalength = datalength
        #####################################################################################

        frame = torch.stack([self.original_data_tensors[i][int(j)] for i,j in zip(random_frame_id,rand_frames)])
        return frame,random_frame_id,rand_frames,datalength

    def get_frame_batch_by_timelist(self, motion_id, frame_num, robot_state) -> torch.Tensor:  
        #按照timelist提供的时间提取4帧动作数据，如下[root_pos_error,root_rot_error,joint_pos]
        frame_parts = []  
        
        for x in self.time_list:  
            frame_step_num = int(x // self.frame_duration)  
            frame = self.get_frame_batch(motion_id, frame_num + frame_step_num)  
            
            # 计算位置误差  
            pos_error = frame[:, :3] - robot_state[:, :3]  
            
            # 计算旋转误差  
            rot_error = quat_error_magnitude(frame[:, 3:7], robot_state[:, 3:7])  
            
            # 拼接 pos_error 和 rot_error  
            frame_part = torch.cat([pos_error, rot_error], dim=1)  # 确保形状匹配  
            
            # 拼接 joint_pos_i  
            joint_pos_i = self.joint_position_w(frame)  
            frame_part = torch.cat([frame_part, joint_pos_i], dim=1)  
            
            # 累加到 frame_parts  
            frame_parts.append(frame_part)  
        
        # 最后一次性拼接  
        return torch.cat(frame_parts, dim=1)
    
    def get_frame_batch(self,motion_id,frame_num):
        """
        根据给定的motion_id和frame_num获取一批帧数据。
        
        Args:
            motion_id (list of int): motion的ID列表。
            frame_num (list of float): 对应motion的帧编号列表。
        
        Returns:
            torch.Tensor: 一个堆叠的张量，包含所有指定motion和帧的数据。
        
        """
        batch = []

        # 遍历 motion_id 和 frame_num
        for i, j in zip(motion_id, frame_num):
            # 将 j 向下取整
            index = int(j)

            # 检查索引是否越界
            if index >= len(self.original_data_tensors[i]):
                index -= 1
                print(f"Warning: Index {index} is out of bounds for motion {i} with length {len(self.original_data_tensors[i])}.")
                continue

            # 添加合法索引的数据到 batch
            batch.append(self.original_data_tensors[i][index])

        # 返回堆叠后的张量
        return torch.stack(batch)
            
    def get_frame_by_header(self, frame, header):
        """
        从给定的二维frame矩阵中，根据header列表返回对应的列。
        """
        # 获取frame的表头
        frame_header = self.headers[0]  # 假设所有文件的表头相同，这里取第一个文件的表头
        column_indices = [frame_header.index(h) for h in header if h in frame_header]
        
        if not column_indices:
            raise ValueError("None of the provided headers match the frame's headers.")
        
        # 选择对应的列
        selected_columns = frame[:, column_indices]
        
        return selected_columns
        
    def _get_states_info_by_interpolation(self, frame_data_c: torch.Tensor, frame_data_n: torch.Tensor, 
                                        frame_frac: float, free_joint=True, interpolation=True):
        if interpolation:
            # 确保输入为二维张量
            assert frame_data_c.dim() == 2 and frame_data_n.dim() == 2, "Inputs must be 2D tensors"
            
            # 如果需要进行插值
            base_pos = self.base_pos_interpolation(frame_data_c[:, :3], frame_data_n[:, :3], frame_frac)
            # 插值计算基本位置
            base_orn = self.base_orn_interpolation(frame_data_c[:, 3:7], frame_data_n[:, 3:7], frame_frac)
            # 插值计算基本方向
            base_lin_vel = self.base_lin_vel_interpolation(frame_data_c[:, :3], frame_data_n[:, :3], self.frame_duration)
            # 插值计算基本线性速度
            base_ang_vel = self.base_ang_vel_interpolation(frame_data_c[:, 3:7], frame_data_n[:, 3:7], self.frame_duration)
            # 插值计算基本角速度
            joint_pos, joint_vel = self.joint_interpolation(frame_data_c[:, 7:], frame_data_n[:, 7:], frame_frac,
                                                            self.frame_duration, free_joint)  
            
            return base_pos, base_orn, base_lin_vel, base_ang_vel, joint_pos, joint_vel

    @staticmethod
    def base_orn_interpolation(base_orn_c: torch.Tensor, base_orn_n: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """
        对给定的四元数进行插值计算。

        参数:
        base_orn_c (torch.Tensor): 当前的四元数张量，形状为(batch_size, 4)。
        base_orn_n (torch.Tensor): 下一个的四元数张量，形状为(batch_size, 4)。
        frac (torch.Tensor): 插值比例因子张量，形状为(batch_size,) 或 (batch_size, 1)。

        返回:
        torch.Tensor: 插值后的四元数张量，形状为(batch_size, 4)。
        """
        # 将四元数转换为轴角表示
        axis_angle_c = axis_angle_from_quat(base_orn_c)
        axis_angle_n = axis_angle_from_quat(base_orn_n)

        # 确保 frac 是形状为 (batch_size, 1) 的张量
        if frac.dim() == 1:
            frac = frac.unsqueeze(1)

        # 对轴角进行线性插值
        axis_angle_interp = axis_angle_c + frac * (axis_angle_n - axis_angle_c)

        # 将插值后的轴角转换回四元数
        base_orn = quat_from_angle_axis(axis_angle_interp.norm(dim=-1), axis_angle_interp / axis_angle_interp.norm(dim=-1, keepdim=True))

        return base_orn

    @staticmethod
    def base_lin_vel_interpolation(base_pos_c: torch.Tensor, base_pos_n: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        计算基础线性速度的插值
        
        通过当前和下一个时间步的位置来计算基础线性速度的变化率，即线速度。
        这个方法主要用于在给定两个时间点的位置时，计算物体在这段时间内的平均线速度。
        
        参数:
        - base_pos_c: 当前时间步的基础位置，形状为 (batch_size, 3)。
        - base_pos_n: 下一个时间步的基础位置，形状为 (batch_size, 3)。
        - delta_t: 两个时间步之间的时间间隔，形状为 (batch_size,) 或标量。
        
        返回:
        - base_lin_vel: 基础线性速度，形状为 (batch_size, 3)。
        """
        # 计算位置差
        pos_diff = base_pos_n - base_pos_c

        # 确保 delta_t 是形状为 (batch_size, 1) 的张量
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0).unsqueeze(1)
        elif delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)

        # 计算线速度
        base_lin_vel = pos_diff / delta_t

        return base_lin_vel

    @staticmethod
    def base_ang_vel_interpolation(base_orn_c: torch.Tensor, base_orn_n: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        计算基础角速度插值。

        该方法用于根据当前和下一个时刻的四元数姿态，计算物体的角速度。
        它通过计算两个姿态之间的旋转矢量来实现这一点，然后基于旋转轴和旋转角度计算角速度。
        参数:
        - base_orn_c: 当前时刻的四元数姿态，形状为 (batch_size, 4)。
        - base_orn_n: 下一个时刻的四元数姿态，形状为 (batch_size, 4)。
        - delta_t: 两个时刻之间的时间差，形状为 (batch_size,) 或标量。

        返回:
        - base_ang_vel: 基础角速度，形状为 (batch_size, 3)。
        """
        # 计算两个姿态之间的相对旋转矢量
        rotvec = quat_error_magnitude(base_orn_c, base_orn_n)
        
        # 计算旋转角度
        angle = torch.norm(rotvec, dim=-1, keepdim=True)

        # 计算旋转轴
        axis = rotvec / (angle + 1e-8)

        # 将 delta_t 转换为形状为 (batch_size, 1) 的张量
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0).unsqueeze(1)
        elif delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)

        # 计算角速度
        base_ang_vel = axis * (angle / delta_t)

        return base_ang_vel

    @staticmethod
    def joint_interpolation(joint_pos_c: torch.Tensor, joint_pos_n: torch.Tensor, frac: torch.Tensor, delta_t: torch.Tensor, free_joint: bool = True) -> (torch.Tensor, torch.Tensor):
        """
        实现关节空间的插值计算。

        根据当前关节位置、目标关节位置和一个时间因子，计算插值后的关节位置和速度。
        如果关节不是自由关节，则添加轮子相关的数据。

        参数:
        - joint_pos_c: 当前的关节位置，形状为 (batch_size, n_joints)。
        - joint_pos_n: 目标的关节位置，形状为 (batch_size, n_joints)。
        - frac: 插值因子，用于混合当前位置和目标位置，形状为 (batch_size,) 或 (batch_size, 1)。
        - delta_t: 时间差，用于计算关节速度，形状为 (batch_size,) 或标量。
        - free_joint: 布尔值，指示是否为自由关节。

        返回:
        - joint_pos: 插值后的关节位置，形状为 (batch_size, n_joints)。
        - joint_vel: 计算得到的关节速度，形状为 (batch_size, n_joints)。
        """

        # 确保 frac 是形状为 (batch_size, 1) 的张量
        if frac.dim() == 1:
            frac = frac.unsqueeze(1)

        # 计算插值后的关节位置
        joint_pos = joint_pos_c + frac * (joint_pos_n - joint_pos_c)

        # 确保 delta_t 是形状为 (batch_size, 1) 的张量
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0).unsqueeze(1)
        elif delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)

        # 计算关节速度
        joint_vel = (joint_pos_n - joint_pos_c) / delta_t

        # 返回插值后的关节位置和速度
        return joint_pos, joint_vel     

    @staticmethod
    def base_pos_interpolation(base_pos_c: torch.Tensor, base_pos_n: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """
        对给定的基础位置进行插值计算。

        参数:
        base_pos_c (torch.Tensor): 当前的基础位置张量，形状为(batch_size, 3)。
        base_pos_n (torch.Tensor): 下一个的基础位置张量，形状为(batch_size, 3)。
        frac (torch.Tensor): 每一行的比例因子张量，形状为(batch_size,) 或 (batch_size, 1)。

        返回:
        torch.Tensor: 插值后的位置张量，形状为(batch_size, 3)。
        """
        # 确保 frac 是形状为 (batch_size, 1) 的张量
        if frac.dim() == 1:
            frac = frac.unsqueeze(1)

        # 计算插值后的基础位置
        base_pos = base_pos_c + frac * (base_pos_n - base_pos_c)
        
        return base_pos   

    @staticmethod
    def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, fraction: torch.Tensor, spin: int = 0, shortestpath: bool = True) -> torch.Tensor:
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

    @staticmethod
    def slerp(self, val0: torch.Tensor, val1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return (1.0 - blend) * val0 + blend * val1   
    
    def get_frames(self,motion_id,frame_num):
        """
        读取数据tensor，返回一个frame
        """
        return self.original_data_tensors[motion_id][frame_num]    
    
    #############################PROPERTY############################
    @property    
    def get_tensors(self):
        """
        返回加载的所有数据的列表。
        
        :return: 包含所有CSV文件数据的二维Tensor列表
        """
        return self.original_data_tensors    
    
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


#######################MATH##############################################
@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """将四元数表示的旋转转换为轴角表示。

    参数:
        quat: 四元数的方向，表示为 (w, x, y, z)。形状为 (..., 4)。
        eps: 泰勒近似的容差。默认值为 1.0e-6。

    返回:
        以轴角形式表示的旋转。形状为 (..., 3)。
        向量的大小表示绕向量方向逆时针旋转的角度（以弧度为单位）。

    参考:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # 修改为接受四元数 [q_w, q_x, q_y, q_z]
    # 四元数 [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # 轴角表示 [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # 因此，轴角表示为 [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # 当 theta = 0 时，(sin(theta/2) / theta) 是未定义的
    # 但是，当 theta 趋近于 0 时，可以使用泰勒近似 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """计算四元数的共轭。

    参数:
        q: 四元数的方向，表示为 (w, x, y, z)。形状为 (..., 4)。

    返回:
        四元数的共轭，表示为 (w, x, y, z)。形状为 (..., 4)。
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """将两个四元数相乘。

    参数:
        q1: 第一个四元数，表示为 (w, x, y, z)。形状为 (..., 4)。
        q2: 第二个四元数，表示为 (w, x, y, z)。形状为 (..., 4)。

    返回:
        两个四元数的乘积，表示为 (w, x, y, z)。形状为 (..., 4)。

    异常:
        ValueError: 输入 `q1` 和 `q2` 的形状不匹配。
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return axis_angle_from_quat(quat_diff)
    
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



