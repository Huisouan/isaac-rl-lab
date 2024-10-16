import glob
import json
import numpy as np
import torch

from pybullet_utils import transformations
from . import motion_util, pose3d
from . import amp_utils

class DATALoader:

    # root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel
    POS_SIZE = 3  # base pos
    ROT_SIZE = 4  # base rot
    LINEAR_VEL_SIZE = 3  # base v
    ANGULAR_VEL_SIZE = 3  # base omega
    JOINT_POS_SIZE = 12  # joint theta
    JOINT_VEL_SIZE = 12  # joint dtheta
    TAR_TOE_POS_LOCAL_SIZE = 12  # foot pos
    TAR_TOE_VEL_LOCAL_SIZE = 12  # foot v

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("data/*"),
    ):
        """初始化ExpertDataset类实例。

        参数:
        - device: 硬件设备，如CPU或特定的GPU。
        - time_between_frames: 过渡之间的秒数。
        - data_dir: 数据目录路径（默认为空）。
        - preload_transitions: 是否预加载转换（默认为False）。
        - num_preload_transitions: 预加载的转换数量（默认为1000000）。
        - motion_files: 动作文件列表，默认为glob.glob("datasets/motion_files2/*")。

        说明:
        该构造函数设置ExpertDataset，用于提供Dog mocap数据集的AMP观测。
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # 存储每个轨迹的值。
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # 轨迹长度，以秒为单位。
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        # 遍历所有运动文件，处理并加载数据。
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                # 将数据重新排序以适应Isaac Gym。
                # motion_data = self.reorder_from_pybullet_to_isaac(motion_data)
                motion_data = self.reorder_from_isaacgym_to_isaacgym(motion_data)
                
                # 标准化和规范化四元数。
                for f_i in range(motion_data.shape[0]):
                    root_rot = DATALoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[f_i, DATALoader.POS_SIZE : (DATALoader.POS_SIZE + DATALoader.ROT_SIZE)] = root_rot

                # 移除前7个观察维度（root_pos和root_orn）。
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, DATALoader.ROOT_ROT_END_IDX : DATALoader.JOINT_VEL_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, : DATALoader.JOINT_VEL_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # 轨迹权重用于比其他轨迹更多地采样一些轨迹。

        self.trajectory_weights = torch.tensor(self.trajectory_weights, dtype=torch.float32, device=self.device) / torch.sum(torch.tensor(self.trajectory_weights, dtype=torch.float32, device=self.device))  
        self.trajectory_frame_durations = torch.tensor(self.trajectory_frame_durations, dtype=torch.float32, device=self.device)  
        self.trajectory_lens = torch.tensor(self.trajectory_lens, dtype=torch.float32, device=self.device)  
        self.trajectory_num_frames = torch.tensor(self.trajectory_num_frames, dtype=torch.float32, device=self.device)

        # 预加载转换。
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)

            print(self.get_joint_pose_batch(self.preloaded_s).mean(dim=0))
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def reorder_from_isaacgym_to_isaacgym_tool(self, joint_tensor):
        # 将 numpy 数组转换为 4x3 数组
        # Convert numpy array to a 4x3 array
        reshaped_array = joint_tensor.reshape(-1, 4, 3)
        # 对数组进行转置
        # Transpose the array
        transposed_array = np.transpose(reshaped_array, (0, 2, 1))
        # 将数组重新展平为一维数组
        # Flatten the array back to 1 dimension
        rearranged_array = transposed_array.reshape(-1, 12)
        # 转换为 torch 张量
        # Convert to torch tensor
        rearranged_tensor = torch.tensor(rearranged_array)
        return rearranged_tensor
        
    def reorder_from_isaacgym_to_isaacgym(self, motion_data):
        """Convert from PyBullet ordering to Isaac ordering.

        # 将腿和关节的顺序从PyBullet的[FR, FL, RR, RL]重新排列为IsaacGym的顺序[FL, FR, RL, RR]。
        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].
        """
        # 获取根节点的位置
        root_pos = DATALoader.get_root_pos_batch(motion_data)
        # 获取根节点的旋转
        root_rot = DATALoader.get_root_rot_batch(motion_data)
        # 将旋转的第四维（w）和前三维（x, y, z）进行拼接，调整维度顺序
        root_rot = np.concatenate((root_rot[:, 3].reshape(-1, 1), root_rot[:, 0:3]), axis=1)

        # 获取关节的位置
        joint_pos = DATALoader.get_joint_pose_batch(motion_data)
        # 调用工具函数进行顺序调整
        joint_pos = self.reorder_from_isaacgym_to_isaacgym_tool(joint_pos)

        # 获取脚趾在局部坐标系下的位置
        foot_pos = DATALoader.get_tar_toe_pos_local_batch(motion_data)
        # 调用工具函数进行顺序调整
        #foot_pos = self.reorder_from_isaacgym_to_isaacgym_tool(foot_pos)

        # 获取线性速度
        lin_vel = DATALoader.get_linear_vel_batch(motion_data)
        # 获取角速度
        ang_vel = DATALoader.get_angular_vel_batch(motion_data)

        # 获取关节速度
        joint_vel = DATALoader.get_joint_vel_batch(motion_data)
        # 调用工具函数进行顺序调整
        joint_vel = self.reorder_from_isaacgym_to_isaacgym_tool(joint_vel)

        # 获取脚趾在局部坐标系下的速度
        foot_vel = DATALoader.get_tar_toe_vel_local_batch(motion_data)
        # 调用工具函数进行顺序调整
        foot_vel = self.reorder_from_isaacgym_to_isaacgym_tool(foot_vel)

        # 将所有结果水平堆叠成一个数组并返回
        return np.hstack([root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel])

    def reorder_from_pybullet_to_isaac(self, motion_data):
        """Convert from PyBullet ordering to Isaac ordering.

        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].
        """
        root_pos = DATALoader.get_root_pos_batch(motion_data)
        root_rot = DATALoader.get_root_rot_batch(motion_data)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(DATALoader.get_joint_pose_batch(motion_data), 4, axis=1)
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])

        fp_fr, fp_fl, fp_rr, fp_rl = np.split(DATALoader.get_tar_toe_pos_local_batch(motion_data), 4, axis=1)
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])

        lin_vel = DATALoader.get_linear_vel_batch(motion_data)
        ang_vel = DATALoader.get_angular_vel_batch(motion_data)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(DATALoader.get_joint_vel_batch(motion_data), 4, axis=1)
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(DATALoader.get_tar_toe_vel_local_batch(motion_data), 4, axis=1)
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        return np.hstack([root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel])

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """
        批量采样轨迹索引。
        
        Args:
            size (int): 采样大小。
            
        Returns:
            torch.Tensor: 采样得到的轨迹索引张量，形状为 (size,)。
        """
        # 确保轨迹索引和权重已经转换为 Tensor
        trajectory_idxs_tensor = torch.as_tensor(self.trajectory_idxs, device=self.device, dtype=torch.long)
        trajectory_weights_tensor = torch.as_tensor(self.trajectory_weights, device=self.device, dtype=torch.float32)
        
        # 使用 torch.multinomial 进行加权采样
        sampled_indices = torch.multinomial(trajectory_weights_tensor, num_samples=size, replacement=True)
        
        # 从轨迹索引中提取采样得到的索引
        sampled_traj_idxs = trajectory_idxs_tensor[sampled_indices]
        
        return sampled_traj_idxs
    
    def traj_time_sample(self, traj_idx):
        """
        对轨迹进行随机时间采样。
        
        Args:
            traj_idx (int): 轨迹索引。
        
        Returns:
            float: 采样得到的随机时间。
        
        """
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """
        从多个轨迹中随机采样时间，确保采样时间不在每个轨迹最后的0.5秒内。
        
        Args:
            traj_idxs (torch.Tensor or np.ndarray): 需要采样时间的轨迹索引数组。
            
        Returns:
            torch.Tensor: 采样得到的时间数组，形状与 `traj_idxs` 相同。
        """
        # 检查并转换输入参数为 Tensor
        if not isinstance(traj_idxs, torch.Tensor):
            traj_idxs = torch.tensor(traj_idxs, device=self.device, dtype=torch.long)

        # 计算 `subst`
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        # 生成随机数
        random_samples = torch.rand(len(traj_idxs), device=self.device, dtype=torch.float32)

        # 计算时间样本
        max_time = self.trajectory_lens[traj_idxs] - 0.5  # 确保时间不超过轨迹长度减去0.5秒
        time_samples = max_time * random_samples - subst

        # 确保时间样本非负
        time_samples = torch.maximum(torch.zeros_like(time_samples, dtype=torch.float32), time_samples)

        return time_samples

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = torch.floor(p * n), torch.ceil(p * n)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """
        根据给定的轨迹索引和时间，返回该轨迹在指定时间下的完整帧。
        
        Args:
            traj_idx (int): 轨迹索引。
            time (float): 时间，取值范围为[0, trajectory_lens[traj_idx]]。
        
        Returns:
            np.ndarray: 完整帧的位姿矩阵，shape为(4, 4)。
        
        """
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(torch.floor(p * n)), int(torch.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """
        获取指定轨迹和时间对应的完整帧。
        
        Args:
            traj_idxs (torch.Tensor or list or numpy.ndarray): 轨迹索引，形状为 (batch_size,)。
            times (torch.Tensor or list or numpy.ndarray): 时间点，形状为 (batch_size,)。
        
        Returns:
            torch.Tensor: 形状为 (batch_size, DATALoader.POS_SIZE + DATALoader.ROT_SIZE + DATALoader.JOINT_VEL_END_IDX - DATALoader.JOINT_POSE_START_IDX) 的张量，
                包含了指定轨迹和时间对应的完整帧信息，包括位置、旋转和关节振幅。
        
        """
        # 检查并转换输入参数为 Tensor
        if not isinstance(traj_idxs, torch.Tensor):
            traj_idxs = torch.tensor(traj_idxs, device=self.device)
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times, device=self.device)

        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = torch.floor(p * n).to(torch.int64)
        idx_high = torch.ceil(p * n).to(torch.int64)
        
        batch_size = len(traj_idxs)
        all_frame_pos_starts = torch.zeros(batch_size, DATALoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(batch_size, DATALoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(batch_size, DATALoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(batch_size, DATALoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(batch_size, DATALoader.JOINT_VEL_END_IDX - DATALoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(batch_size, DATALoader.JOINT_VEL_END_IDX - DATALoader.JOINT_POSE_START_IDX, device=self.device)
        
        unique_traj_idxs = torch.unique(traj_idxs)
        for traj_idx in unique_traj_idxs:
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = DATALoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = DATALoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = DATALoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = DATALoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, DATALoader.JOINT_POSE_START_IDX:DATALoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, DATALoader.JOINT_POSE_START_IDX:DATALoader.JOINT_VEL_END_IDX]
        
        blend = (p * n - idx_low).unsqueeze(-1).to(self.device, torch.float32)


        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = amp_utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        
        

        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """
        返回随机完整帧。
        
        Args:
            无参数。
        
        Returns:
            np.ndarray: 随机完整帧的numpy数组。
        
        """
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_envs):
        """
        根据给定的帧数，获取一批完整的帧。
        
        Args:
            num_frames (int): 需要获取的帧数。
        
        Returns:
            np.ndarray: 一个形状为 (num_envs, height, width, channels) 的数组，
                        其中包含了一批完整的帧。
        
        """
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_envs)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_envs)
            times = self.traj_time_sample_batch(traj_idxs)            
            return self.get_full_frame_at_time_batch(traj_idxs, times),times,traj_idxs

    def blend_frame_pose(self, frame0, frame1, blend):
        """
        在两个帧之间进行线性插值，包括方向。
        
        Args:
            frame0: 第一帧，对应于blend = 0。
            frame1: 第二帧，对应于blend = 1。
            blend: 浮点数，取值范围在[0, 1]之间，指定两个帧之间的插值系数。
        
        Returns:
            两个帧的插值结果。
        
        """
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = DATALoader.get_root_pos(frame0), DATALoader.get_root_pos(frame1)
        root_rot0, root_rot1 = DATALoader.get_root_rot(frame0), DATALoader.get_root_rot(frame1)
        joints0, joints1 = DATALoader.get_joint_pose(frame0), DATALoader.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = DATALoader.get_tar_toe_pos_local(frame0), DATALoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = DATALoader.get_linear_vel(frame0), DATALoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = DATALoader.get_angular_vel(frame0), DATALoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = DATALoader.get_joint_vel(frame0), DATALoader.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device
        )
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat(
            [
                blend_root_pos,
                blend_root_rot,
                blend_joints,
                blend_tar_toe_pos,
                blend_linear_vel,
                blend_angular_vel,
                blend_joints_vel,
            ]
        )

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, DATALoader.JOINT_POSE_START_IDX : DATALoader.JOINT_VEL_END_IDX]
                s = torch.cat(
                    [s, self.preloaded_s[idxs, DATALoader.ROOT_POS_START_IDX + 2 : DATALoader.ROOT_POS_START_IDX + 3]],
                    dim=-1,
                )
                s_next = self.preloaded_s_next[idxs, DATALoader.JOINT_POSE_START_IDX : DATALoader.JOINT_VEL_END_IDX]
                s_next = torch.cat(
                    [
                        s_next,
                        self.preloaded_s_next[
                            idxs, DATALoader.ROOT_POS_START_IDX + 2 : DATALoader.ROOT_POS_START_IDX + 3
                        ],
                    ],
                    dim=-1,
                )
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] + 1

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @staticmethod
    def get_root_pos(pose):
        return pose[DATALoader.ROOT_POS_START_IDX : DATALoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, DATALoader.ROOT_POS_START_IDX : DATALoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        return pose[DATALoader.ROOT_ROT_START_IDX : DATALoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, DATALoader.ROOT_ROT_START_IDX : DATALoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose(pose):
        return pose[DATALoader.JOINT_POSE_START_IDX : DATALoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, DATALoader.JOINT_POSE_START_IDX : DATALoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_tar_toe_pos_local(pose):
        return pose[DATALoader.TAR_TOE_POS_LOCAL_START_IDX : DATALoader.TAR_TOE_POS_LOCAL_END_IDX]

    @staticmethod
    def get_tar_toe_pos_local_batch(poses):
        return poses[:, DATALoader.TAR_TOE_POS_LOCAL_START_IDX : DATALoader.TAR_TOE_POS_LOCAL_END_IDX]

    @staticmethod
    def get_linear_vel(pose):
        return pose[DATALoader.LINEAR_VEL_START_IDX : DATALoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_linear_vel_batch(poses):
        return poses[:, DATALoader.LINEAR_VEL_START_IDX : DATALoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel(pose):
        return pose[DATALoader.ANGULAR_VEL_START_IDX : DATALoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, DATALoader.ANGULAR_VEL_START_IDX : DATALoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        return pose[DATALoader.JOINT_VEL_START_IDX : DATALoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, DATALoader.JOINT_VEL_START_IDX : DATALoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_tar_toe_vel_local(pose):
        return pose[DATALoader.TAR_TOE_VEL_LOCAL_START_IDX : DATALoader.TAR_TOE_VEL_LOCAL_END_IDX]

    @staticmethod
    def get_tar_toe_vel_local_batch(poses):
        return poses[:, DATALoader.TAR_TOE_VEL_LOCAL_START_IDX : DATALoader.TAR_TOE_VEL_LOCAL_END_IDX]
'''
import os
import glob
import torch
import csv

class DataLoader:
    def __init__(self, csv_path: str, device: str = "cuda"):
        self.frameduration = 0.005
        self.frames = self.read_csv(csv_path)
        self.device = device
        

    def read_csv(self, csv_path: str):
        """
        Reads all CSV files in the given directory and returns a tensor of shape (num_frames, num_joints, 3).
        If a cell contains NaN, it replaces the value with the average of the values above and below it.
        """
        # 获取文件夹中所有的 CSV 文件路径
        csv_files = glob.glob(os.path.join(csv_path, "*.csv"))

        # 初始化一个空列表来存储所有 CSV 文件的数据
        all_data = []

        for file_path in csv_files:
            # 使用 csv 模块读取 CSV 文件
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过表头
                data = [list(map(float, row)) for row in reader]

            # 将列表转换为 PyTorch 张量
            tensor_data = torch.tensor(data, dtype=torch.float32)

            # 处理 NaN 值
            tensor_data = self._replace_nans(tensor_data)

            # 将当前 CSV 文件的数据添加到列表中
            all_data.append(tensor_data)

        # 如果需要将所有数据合并成一个大张量，可以使用 torch.cat 或者保持为列表形式
        if len(all_data) > 0:
            return all_data.to(self.device)
        else:
            return None

    def _replace_nans(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Replaces NaN values in a tensor with the average of the values above and below it.
        """
        # 确保 tensor 是二维的
        assert tensor.dim() == 2, "Tensor must be two-dimensional."

        # 检查每列是否有 NaN 值，并进行替换
        for col_idx in range(tensor.shape[1]):
            column = tensor[:, col_idx]
            nan_indices = torch.isnan(column)

            # 如果该列没有 NaN 值，跳过此列
            if not torch.any(nan_indices):
                continue

            # 查找连续的非 NaN 值
            prev_non_nan = None
            next_non_nan = None
            for i, is_nan in enumerate(nan_indices):
                if is_nan:
                    if prev_non_nan is not None and next_non_nan is not None:
                        # 替换 NaN 值
                        tensor[i, col_idx] = (column[prev_non_nan] + column[next_non_nan]) / 2.0
                else:
                    if prev_non_nan is None:
                        prev_non_nan = i
                    next_non_nan = i

            # 处理边界情况
            if prev_non_nan is not None:
                first_nan_index = nan_indices.nonzero(as_tuple=True)[0]
                if len(first_nan_index) > 0:
                    tensor[first_nan_index[0], col_idx] = column[prev_non_nan]

            last_non_nan_index = nan_indices.nonzero(as_tuple=True)[0]
            if len(last_non_nan_index) > 0:
                tensor[last_non_nan_index[-1], col_idx] = column[next_non_nan]

        return tensor

    def get_frame_bytime(self, traj_idx,time: float):
        """
        Returns the frame at the given time.
        """
        return self.frames[traj_idx][int(time / self.frameduration)].to(self.device)
        
    def get_frame_byindex(self, traj_idx,index: int):
        """
        Returns the frame at the given index.
        """
        return self.frames[traj_idx][index].to(self.device)
    
    def get_rootstate(self, index: int):
        """
        Returns the root state at the given index.
        """
        return self.frames[index][0:7].to(self.device)

    def get_randombatch(self,envs):
        """
        Returns a random batch of frames from the given frames tensor, allowing repeated selections.

        Parameters:
        - frames: A tensor of shape (num_envs, num_frames, ...).
        - envs: The number of environments.
        - batch_size: The size of the batch to return.

        Returns:
        - A tensor of shape (batch_size, ...) containing randomly selected frames.
        """
        # 获取每个环境中的帧数
        num_frames_per_env = frames.shape[1]
        
        # 随机选择环境和帧索引
        env_indices = torch.randint(low=0, high=envs, size=(envs,))
        frame_indices = torch.randint(low=0, high=num_frames_per_env, size=(batch_size,))
        
        # 根据索引选择帧
        batch = frames[env_indices, frame_indices]
        
        return batch
        
'''