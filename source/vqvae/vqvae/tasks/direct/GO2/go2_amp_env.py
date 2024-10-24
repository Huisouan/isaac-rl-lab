# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


from .common import VecEnvStepReturn
from omni.isaac.lab_assets.ant import ANT_CFG
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, Articulation
from omni.isaac.lab.envs import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from ....datasets.go2_model import GO2_MARKER_CFG,UNITREE_GO2_CFG
from omni.isaac.lab.sensors import ContactSensor,ContactSensorCfg
import math
import torch
from collections.abc import Sequence
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import sample_uniform,matrix_from_quat
from omni.isaac.lab.managers import SceneEntityCfg
import glob
from ....datasets.motionload import MotionData
from omni.isaac.lab.utils.math import quat_rotate,compute_pose_error
@configclass
class AMPEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 1
    num_actions = 12
    num_observations = 207
    num_states = 0
    action_scale =1
    
    kp = 50.0
    kd = 0.5
    
    
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=5.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker: ArticulationCfg = GO2_MARKER_CFG.replace(prim_path="/World/envs/env_.*/Marker")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


class AMPEnv(DirectRLEnv):
    cfg: AMPEnvCfg

    def __init__(self, cfg: AMPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = cfg.action_scale
        
        self.motiondata = MotionData("source/vqvae/data")
        #初始化数据集采样的时间
        self.amp_data_frameinplay = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.amp_data_maxtime = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.amp_data_selected = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.int)
        #记录观察值的缓存
        self.last_observation = None
        self.second_last_observation = None
        self.observation = None
        
        
        self.data = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.robot_foot_id = self.robot.find_bodies(['FL_foot','FR_foot','RL_foot','RR_foot'])         
        
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.marker = Articulation(self.cfg.marker)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        
    def _pd_control(self,):
        tgt_joint_vel = torch.zeros_like(self.actions)
        #这里为了不更改action的输出值，因此新增一个变量
        self.pdaction = self.cfg.kp*(self.actions-self.robot.data.joint_pos) + self.cfg.kd*(tgt_joint_vel - self.robot.data.joint_vel)

    def _apply_action(self) -> None:
        # apply action
        #self._pd_control()
        #self.robot.set_joint_effort_target(self.pdaction) 
        self.robot.set_joint_position_target(self.actions)
        #self.robot.set_joint_position_target(self.actions)
        #update marker
        frame = self.motiondata.get_frame_batch(self.amp_data_selected,self.amp_data_frameinplay)
        rootstate = self.motiondata.root_state_w(frame)
        rootstate[:,:2] = rootstate[:,:2] + self.scene.env_origins[:,:2]
        self.marker.write_root_state_to_sim(rootstate)
        self.marker.write_joint_state_to_sim(self.motiondata.joint_position_w(frame),self.motiondata.joint_velocity_w(frame))
    def _get_observations(self) -> dict:
        joint_state = self.robot.data.root_state_w 
        z_vector = matrix_from_quat(self.robot.data.root_state_w[:, 3:7])[:,2,:]
        obs = torch.cat(#42 in total
            (
                
                self.robot.data.joint_pos,#12
                self.robot.data.joint_vel,#12
                joint_state[:,7:],#13,include[pos， quat， lin_vel， ang_vel]
                z_vector,
                self.actions,#12
            ),
            dim=-1,
        )
        #在第一次调用时，初始化last_observation和second_last_observation
        if self.last_observation is None:
            self.last_observation = obs
        if self.second_last_observation is None:
            self.second_last_observation = obs
        
        robot_state = self.robot.data.root_state_w
        robot_state[:, :2] = robot_state[:, :2] - self.scene.env_origins[:, :2]
        
        # 获取数据集
        dataset = self.motiondata.get_frame_batch_by_timelist(#64
            self.amp_data_selected,
            self.amp_data_frameinplay,
            robot_state
        )
        observation = torch.cat([self.last_observation,self.second_last_observation,obs,dataset], dim=1)
        # 将数据集和当前状态拼接起来，作为观察值
        observations = {"policy": observation}
        # 更新last_observation和second_last_observation
        self.second_last_observation = self.last_observation
        self.last_observation = obs
    
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        reference_frame = self.motiondata.get_frame_batch(
        self.amp_data_selected, 
        self.amp_data_frameinplay)  
        # base position
        root_state = self.motiondata.root_state_w(reference_frame)
        root_state[:, :2] = root_state[:, :2] + self.scene.env_origins[:, :2]  
        root_pos = root_state[:, :3] 

        root_orn = root_state[:, 3:7]
        # 基础速度
        # base velocities
        lin_vel = root_state[:,7:10]
        ang_vel = root_state[:,10:13]
        #joint position
        joint_pos = self.motiondata.joint_position_w(reference_frame)
        joint_vel = self.motiondata.joint_velocity_w(reference_frame)   
        
        #foot_position
        foot_positions = self.motiondata.foot_position_w(reference_frame)
        foot_positions[:,0:2] += self.scene.env_origins[:, :2]
        foot_positions[:,3:5] += self.scene.env_origins[:, :2]
        foot_positions[:,6:8] += self.scene.env_origins[:, :2]
        foot_positions[:,9:11] += self.scene.env_origins[:, :2]
                
        foot_velocities = self.motiondata.toe_velocity_w(reference_frame)
        
        
        root_pos_error,root_quat_error = compute_pose_error(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, root_pos, root_orn
        )
        (joint_angle_reward,
            joint_velocity_reward ,
            root_position_reward ,
            root_velocity_reward,
            tracking_position_key_body_reward) = compute_rewards(
            lin_vel = lin_vel, 
            ang_vel = ang_vel, 
            joint_pos = joint_pos, 
            joint_vel = joint_vel,
            root_pos_error = root_pos_error,
            root_quat_error = root_quat_error,
            foot_positions=foot_positions,
            foot_velocities=foot_velocities,
            robot_joint_pos = self.robot.data.joint_pos, 
            robot_joint_vel= self.robot.data.joint_vel,
            robot_root_lin_vel_w= self.robot.data.root_lin_vel_w,
            robot_root_ang_vel_w = self.robot.data.root_ang_vel_w,
            robot_foot_positions = self.robot.data.body_pos_w[:,self.robot_foot_id[0],:].view(self.num_envs,-1),      
            robot_foot_velocities = self.robot.data.body_vel_w[:,self.robot_foot_id[0],:].view(self.num_envs,-1) ,            
        )
        self.root_position_reward = root_position_reward
        self.tracking_position_key_body_reward = tracking_position_key_body_reward
        total_reward = joint_angle_reward + joint_velocity_reward + root_position_reward + root_velocity_reward + tracking_position_key_body_reward
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        #threshold for reward threshold
        root_threshold = 0.00001
        run_out_of_data = self.amp_data_frameinplay >= self.amp_data_maxtime
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        outrangeroot = self.root_position_reward<root_threshold
        #foot_out = self.tracking_position_key_body_reward<tracking_position_key_body_reward_threshold
        return outrangeroot,time_out|run_out_of_data

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
            
        super()._reset_idx(env_ids)
        
        frames,data_idx,rand_frame,data_length = self.motiondata.get_random_frame_batch(len(env_ids))
        root_state = self.motiondata.root_state_w(frames)
        root_state[:, :2] = root_state[:, :2] + self.scene.env_origins[env_ids, :2]        

        #joint position
        joint_pos = self.motiondata.joint_position_w(frames)
        joint_vel = self.motiondata.joint_velocity_w(frames)
        
        # 将值设置到物理仿真中
        # set into the physics simulation
        self.robot.write_root_state_to_sim(root_state,env_ids=env_ids)
        # set into the physics simulation
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids) 
        self.amp_data_frameinplay[env_ids] = rand_frame.to(device=self.device).to(dtype=torch.float)
        self.amp_data_selected[env_ids] = data_idx.to(device=self.device).to(dtype=torch.int)
        self.amp_data_maxtime[env_ids] = data_length.to(device=self.device).to(dtype=torch.float)
        
        #更新observations
        joint_state = self.robot.data.root_state_w 
        joint_state[:,:2]-= self.scene.env_origins[:,:2]        
        
        self.second_last_observation = self.last_observation = None

        
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        执行环境动力学的一个时间步。

        环境以固定的时间步前进，而物理模拟则以较低的时间步进行离散化。这是为了确保模拟的稳定性。
        这两个时间步可以独立配置，使用 :attr:`DirectRLEnvCfg.decimation`（每个环境步骤的模拟步数）
        和 :attr:`DirectRLEnvCfg.sim.physics_dt`（物理时间步）。基于这些参数，环境时间步计算为两者的乘积。

        此函数执行以下步骤：

        1. 在进行物理步骤之前预处理动作。
        2. 将动作应用于模拟器，并以离散化的方式进行物理步骤。
        3. 计算奖励和结束信号。
        4. 重置已终止或达到最大剧集长度的环境。
        5. 如果启用了间隔事件，则应用间隔事件。
        6. 计算观测值。

        参数:
            action: 应用于环境的动作。形状为 (num_envs, action_dim)。

        返回:
            包含观测值、奖励、重置信号（终止和截断）和额外信息的元组。
        """
        action = action.to(self.device)
        # 添加动作噪声

        # 处理动作
        self._pre_physics_step(action)

        # 检查是否需要在物理循环中进行渲染
        # 注意：在此处一次检查以避免在循环中多次检查
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # 执行物理步骤
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # 将动作设置到缓冲区
            self._apply_action()
            # 将动作设置到模拟器
            self.scene.write_data_to_sim()
            # 模拟
            self.sim.step(render=False)
            # 只有在GUI或RTX传感器需要的情况下，在步骤之间进行渲染
            # 注意：我们假设渲染间隔是最短的接受渲染间隔。
            #      如果相机需要更频繁的渲染，这将导致意外行为。
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # 在模拟时间步更新缓冲区
            self.scene.update(dt=self.physics_dt)
            
        # 后处理步骤：
        # -- 更新环境计数器（用于生成课程）
        self.episode_length_buf += 1  # 当前剧集中的步骤（每环境）
        self.common_step_counter += 1  # 总步骤（所有环境共用）

        self.reward_buf = self._get_rewards()
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        # -- 重置已终止或超时的环境，并记录剧集信息
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.sim.render()

        # 后处理步骤：执行间隔事件
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 更新观测值
        self.obs_buf = self._get_observations()
        self.amp_data_frameinplay +=2
        #将数据集更新
        # 添加观测噪声
        # 注意：我们不对状态空间应用噪声（因为它用于批评网络）
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # 返回观测值、奖励、重置信号和额外信息
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


@torch.jit.script
def compute_rewards(            

            lin_vel,
            ang_vel,
            joint_pos,
            joint_vel,
            foot_positions,
            foot_velocities,
            robot_joint_pos,
            robot_joint_vel,
            root_pos_error,
            root_quat_error ,
            robot_root_lin_vel_w,
            robot_root_ang_vel_w,
            robot_foot_positions,
            robot_foot_velocities,
            ):
    # clamp joint pos to limits
    """
    joint_pos_limits = data.soft_joint_pos_limits
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    # clamp joint vel to limits
    joint_vel_limits = data.soft_joint_vel_limits
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)   
    """
    # 将所有张量转换为64位浮点数
    """
    robot_joint_pos = robot_joint_pos.to(torch.float64)
    joint_pos = joint_pos.to(torch.float64)
    robot_joint_vel = robot_joint_vel.to(torch.float64)
    joint_vel = joint_vel.to(torch.float64)
    robot_foot_positions = robot_foot_positions.to(torch.float64)
    foot_positions = foot_positions.to(torch.float64)
    root_pos_error = root_pos_error.to(torch.float64)
    root_quat_error = root_quat_error.to(torch.float64)
    robot_root_lin_vel_w = robot_root_lin_vel_w.to(torch.float64)
    lin_vel = lin_vel.to(torch.float64)
    robot_root_ang_vel_w = robot_root_ang_vel_w.to(torch.float64)
    ang_vel = ang_vel.to(torch.float64)
    """
    # 使用64位浮点数进行计算
    joint_angle_reward = 0.6 * torch.exp(-1 * torch.sum((robot_joint_pos - joint_pos) ** 2, dim=1))
    joint_vel_sum = -0.1 * torch.sum((robot_joint_vel - joint_vel) ** 2, dim=1)
    joint_velocity_reward = 0.1 * torch.exp(joint_vel_sum)

    bodyposition_l2 = -40 * torch.sum((robot_foot_positions - foot_positions) ** 2, dim=1)
    tracking_position_key_body_reward = 0.1 * torch.exp(bodyposition_l2)

    root_pos_error_sum = torch.sum(root_pos_error ** 2, dim=1)
    root_quat_error_sum = torch.sum(root_quat_error ** 2, dim=1)

    # 计算指数部分
    exp_part = -20 * root_pos_error_sum - 10 * root_quat_error_sum
    # 计算最终结果
    root_position_reward = 0.15 * torch.exp(exp_part)

    root_velocity_reward = 0.1 * torch.exp(
        -2 * torch.sum((robot_root_lin_vel_w - lin_vel) ** 2, dim=1)
        -0.2 * torch.sum((robot_root_ang_vel_w - ang_vel) ** 2, dim=1)
    )
    
    return (joint_angle_reward,
            joint_velocity_reward ,
            root_position_reward ,
            root_velocity_reward,
            tracking_position_key_body_reward)
