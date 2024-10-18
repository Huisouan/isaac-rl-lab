import os
import csv
import torch
#from omni.isaac.lab.utils.math import *
#from omni.isaac.lab.utils.math import axis_angle_from_quat ,quat_from_angle_axis,quat_error_magnitude
class MotionData:
    def __init__(self, data_dir,frame_duration = 1/120):
        """
        初始化方法，设置数据目录。
        
        :param data_dir: 包含CSV文件的目录路径
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_dir = data_dir
        self.frame_duration = frame_duration
        self.data_tensors = []
        self.data_names = []
        self.data_length = []
        self.data_time_length = []
        self.data_header = []
        #最后的121帧为了不超出范围，不会在初始化的时候被载入
        self.motion_bias = 122
        self.time_list = [1. / 30., 1. / 15., 1. / 3., 1.]
        self.load_data()
        print('Motion Data Loaded Successfully')
    def load_data(self):
        """
        从指定目录加载所有CSV文件，并将每个文件转换为一个不包含表头的二维Tensor。
        """
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
                    self.data_tensors.append(tensor_data)
        self.data_length = torch.tensor(self.data_length, dtype=torch.int64).to(self.device)
        self.data_time_length = torch.tensor(self.data_length * self.frame_duration, dtype=torch.float64).to(self.device)
    def get_tensors(self):
        """
        返回加载的所有数据的列表。
        
        :return: 包含所有CSV文件数据的二维Tensor列表
        """
        return self.data_tensors

    def get_frames(self,motion_id,frame_num):
        return self.data_tensors[motion_id][frame_num]
    
    def get_random_frame_batch(self,batch_size):
        random_frame_id = torch.randint(0,len(self.data_tensors),size=(batch_size,)).to(self.device)
        random_frame_index = torch.rand((1,len(random_frame_id)))[0].to(self.device)
        datalength = self.data_length[random_frame_id]-self.motion_bias
        rand_frames = datalength*random_frame_index
        #####################################################################################
        # 保存随机帧的索引的第二方法
        self.random_frame_id = random_frame_id
        self.rand_frames = rand_frames
        self.datalength = datalength
        #####################################################################################

        frame = torch.stack([self.data_tensors[i][int(j)] for i,j in zip(random_frame_id,rand_frames)])
        return frame,random_frame_id,rand_frames,datalength

    def get_frame_batch_by_timelist(self, motion_id, frame_num, robot_state) -> torch.Tensor:  
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
        batch = []

        # 遍历 motion_id 和 frame_num
        for i, j in zip(motion_id, frame_num):
            # 将 j 向下取整
            index = int(j)

            # 检查索引是否越界
            if index >= len(self.data_tensors[i]):
                index -= 1
                print(f"Warning: Index {index} is out of bounds for motion {i} with length {len(self.data_tensors[i])}.")
                continue

            # 添加合法索引的数据到 batch
            batch.append(self.data_tensors[i][index])

        # 返回堆叠后的张量
        return torch.stack(batch)
        
        
    def get_frame_by_time(self,motion_id,time):
        frame_num = time//self.frame_duration
        
        
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
    def base_orn_interpolation(base_orn_c, base_orn_n, frac):
        """
        对给定的四元数进行插值计算。

        参数:
        base_orn_c (torch.Tensor): 当前的四元数张量，形状为(batch_size, 4)。
        base_orn_n (torch.Tensor): 下一个的四元数张量，形状为(batch_size, 4)。
        frac (torch.Tensor): 插值比例因子张量，形状为(batch_size,)。

        返回:
        torch.Tensor: 插值后的四元数张量，形状为(batch_size, 4)。
        """
        # 将四元数转换为轴角表示
        axis_angle_c = axis_angle_from_quat(base_orn_c)
        axis_angle_n = axis_angle_from_quat(base_orn_n)

        # 确保 frac 是形状为 (batch_size, 1) 的张量
        frac = frac.view(-1, 1)

        # 对轴角进行线性插值
        axis_angle_interp = axis_angle_c + frac * (axis_angle_n - axis_angle_c)

        # 将插值后的轴角转换回四元数
        base_orn = quat_from_angle_axis(axis_angle_interp.norm(dim=-1), axis_angle_interp / axis_angle_interp.norm(dim=-1, keepdim=True))

        return base_orn

    @staticmethod
    def base_lin_vel_interpolation(base_pos_c: torch.Tensor, base_pos_n: torch.Tensor, delta_t) -> torch.Tensor:
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
        delta_t = delta_t.view(-1, 1)

        # 计算线速度
        base_lin_vel = pos_diff / delta_t

        return base_lin_vel

    @staticmethod
    def base_ang_vel_interpolation(base_orn_c: torch.Tensor, base_orn_n: torch.Tensor, delta_t) -> torch.Tensor:
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
        delta_t = delta_t.view(-1, 1)

        # 计算角速度
        base_ang_vel = axis * (angle / delta_t)

        return base_ang_vel

    @staticmethod
    def joint_interpolation(joint_pos_c: torch.Tensor, joint_pos_n: torch.Tensor, frac: float, delta_t, free_joint: bool = True) -> (torch.Tensor, torch.Tensor):
        """
        实现关节空间的插值计算。

        根据当前关节位置、目标关节位置和一个时间因子，计算插值后的关节位置和速度。
        如果关节不是自由关节，则添加轮子相关的数据。

        参数:
        - joint_pos_c: 当前的关节位置，形状为 (batch_size, n_joints)。
        - joint_pos_n: 目标的关节位置，形状为 (batch_size, n_joints)。
        - frac: 插值因子，用于混合当前位置和目标位置。
        - delta_t: 时间差，用于计算关节速度，形状为 (batch_size,) 或标量。
        - free_joint: 布尔值，指示是否为自由关节。

        返回:
        - joint_pos: 插值后的关节位置，形状为 (batch_size, n_joints)。
        - joint_vel: 计算得到的关节速度，形状为 (batch_size, n_joints)。
        """

        # 计算插值后的关节位置
        joint_pos = joint_pos_c + frac * (joint_pos_n - joint_pos_c)

        # 计算关节速度
        delta_t = delta_t.view(-1, 1)

        joint_vel = (joint_pos_n - joint_pos_c) / delta_t

        # 如果不是自由关节，则添加轮子数据
        if not free_joint:
            # 定义添加轮子数据的函数
            def _add_wheel(data: torch.Tensor) -> torch.Tensor:
                zeros = torch.zeros((data.shape[0], 1), dtype=data.dtype, device=data.device)
                return torch.cat([data[:, :3], zeros, data[:, 3:6], zeros, data[:, 6:9], zeros, data[:, 9:12], zeros], dim=1)

            # 对关节位置和速度添加轮子数据
            joint_pos = _add_wheel(joint_pos)
            joint_vel = _add_wheel(joint_vel)

        # 返回插值后的关节位置和速度
        return joint_pos, joint_vel     
        
    @staticmethod
    def base_pos_interpolation(base_pos_c, base_pos_n, frac):
        """
        对给定的基础位置进行插值计算。

        参数:
        base_pos_c (torch.Tensor): 当前的基础位置张量，形状为(batch_size, 3)。
        base_pos_n (torch.Tensor): 下一个的基础位置张量，形状为(batch_size, 3)。
        frac (torch.Tensor): 每一行的比例因子张量，形状为(batch_size,)。

        返回:
        torch.Tensor: 插值后的位置张量，形状为(batch_size, 3)。
        """
        # 将frac扩展为(batch_size, 1)，以便能够进行广播
        frac = frac.view(-1, 1)

        # 计算插值后的基础位置
        base_pos = base_pos_c + frac * (base_pos_n - base_pos_c)
        
        return base_pos   
        
    @staticmethod
    def root_state_w(frame: torch.Tensor) -> torch.Tensor:
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        if frame.dim() == 1:
            return frame[:13]
        elif frame.dim() == 2:
            return frame[:, :13]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    @staticmethod
    def joint_position_w(frame: torch.Tensor) -> torch.Tensor:
        """
        提取关节位置向量。
        
        :param frame: 一维或二维张量
        :return: 关节位置向量
        """
        if frame.dim() == 1:
            return frame[13:25]
        elif frame.dim() == 2:
            return frame[:, 13:25]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    @staticmethod
    def joint_velocity_w(frame: torch.Tensor) -> torch.Tensor:
        """
        提取关节速度向量。
        
        :param frame: 一维或二维张量
        :return: 关节速度向量
        """
        if frame.dim() == 1:
            return frame[25:37]
        elif frame.dim() == 2:
            return frame[:, 25:37]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    @staticmethod
    def toe_position_w(frame: torch.Tensor) -> torch.Tensor:
        """
        提取脚趾位置向量。
        
        :param frame: 一维或二维张量
        :return: 脚趾位置向量
        """
        if frame.dim() == 1:
            return frame[37:49]
        elif frame.dim() == 2:
            return frame[:, 37:49]
        else:
            raise ValueError('Input tensor must be either one or two dimensional.')

    @staticmethod
    def toe_velocity_w(frame: torch.Tensor) -> torch.Tensor:
        """
        提取脚趾速度向量。
        
        :param frame: 一维或二维张量
        :return: 脚趾速度向量
        """
        if frame.dim() == 1:
            return frame[49:]
        elif frame.dim() == 2:
            return frame[:, 49:]
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
    
    


    
    
@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
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
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
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
    
    
    
