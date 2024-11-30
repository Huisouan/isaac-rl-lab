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
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip
from rl_lab.assets.go2_model import GO2_MARKER_CFG
from omni.isaac.lab.sensors import ContactSensor,ContactSensorCfg
import math
import torch
from collections.abc import Sequence
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.managers import SceneEntityCfg
import glob
from rl_lab.assets.base_motionloader import MotionData_Base
from omni.isaac.lab.utils.math import quat_rotate,compute_pose_error
@configclass
class EPMCEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.5
    num_actions = 12
    num_observations = 294
    num_states = 0
    action_scale =1
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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


class EPMCEnv(DirectRLEnv):
    cfg: EPMCEnvCfg

    def __init__(self, cfg: EPMCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = 1

        self.motiondata = MotionData_Base("source/Mycode/data")
        #初始化数据集采样的时间
        self.pmc_data_frameinplay = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.pmc_data_maxtime = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.pmc_data_selected = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.int)
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

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)
        
        #update marker
        frame = self.motiondata.get_frame_batch(self.pmc_data_selected,self.pmc_data_frameinplay)
        rootstate = self.motiondata.root_state_w(frame)
        rootstate[:,:2] = rootstate[:,:2] + self.scene.env_origins[:,:2]
        self.marker.write_root_state_to_sim(rootstate)
        self.marker.write_joint_state_to_sim(self.motiondata.joint_position_w(frame),self.motiondata.joint_velocity_w(frame))
    def _get_observations(self) -> dict:
        joint_state = self.robot.data.root_state_w 
        joint_state[:,:2]-= self.scene.env_origins[:,:2]
        obs = torch.cat(#49 in total
            (
                joint_state,#13,include[pos， quat， lin_vel， ang_vel]
                
                self.robot.data.joint_pos,#12
                self.robot.data.joint_vel,#12
                self.actions,#12
            ),
            dim=-1,
        )
        #在第一次调用时，初始化last_observation和second_last_observation
        if self.last_observation is None:
            self.last_observation = obs
        if self.second_last_observation is None:
            self.second_last_observation = obs
            
        # 获取数据集
        dataset = torch.cat((#147 in total
            self.motiondata.get_frame_batch(self.pmc_data_selected,self.pmc_data_frameinplay+5),
            self.motiondata.get_frame_batch(self.pmc_data_selected,self.pmc_data_frameinplay+10),
            self.motiondata.get_frame_batch(self.pmc_data_selected,self.pmc_data_frameinplay+50)),
            dim=1)    
        observation = torch.cat([self.last_observation,self.second_last_observation,obs,dataset], dim=1)
        # 将数据集和当前状态拼接起来，作为观察值
        observations = {"policy": observation}
        # 更新last_observation和second_last_observation
        self.second_last_observation = self.last_observation
        self.last_observation = obs
    
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        return self.done_buf, self.truncated_buf
    
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
        self.pmc_data_frameinplay[env_ids] = rand_frame.to(device=self.device).to(dtype=torch.float)
        self.pmc_data_selected[env_ids] = data_idx.to(device=self.device).to(dtype=torch.int)
        self.pmc_data_maxtime[env_ids] = data_length.to(device=self.device).to(dtype=torch.float)
        
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
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

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
            #将数据集更新
            self.pmc_data_frameinplay +=1

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
            # 如果场景中添加了传感器，请确保渲染以反映重置的变化
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # 后处理步骤：执行间隔事件
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 更新观测值
        self.obs_buf = self._get_observations()

        # 添加观测噪声
        # 注意：我们不对状态空间应用噪声（因为它用于批评网络）
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # 返回观测值、奖励、重置信号和额外信息
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    import torch

#@torch.jit.script
def _compute_common_reward(
    base_pos: list, 
    base_lin_vel: list, 
    base_orn: list, 
    target_pos: torch.Tensor, 
    total_spd: float, 
    max_spd: float, 
    last_pos_diff_len: float, 
    max_steps: int, 
    reward_rotation: float, 
    reward_dist: float
) :
    """
    计算共同奖励，包括速度奖励、旋转奖励和距离奖励。

    :param base_pos: 当前位置列表 [x, y, z]。
    :param base_lin_vel: 当前线速度列表 [vx, vy, vz]。
    :param base_orn: 当前姿态四元数列表 [x, y, z, w]。
    :param target_pos: 目标位置的张量。
    :param total_spd: 当前累计的速度。
    :param max_spd: 当前最大速度。
    :param last_pos_diff_len: 上一次的位置差长度。
    :param max_steps: 最大步数。
    :param reward_rotation: 当前旋转奖励。
    :param reward_dist: 当前距离奖励。
    :return: 计算后的奖励值，更新后的总速度，最大速度，当前位置差长度，更新后的旋转奖励，更新后的距离奖励。
    """

    # 速度奖励
    current_position = torch.tensor(base_pos, dtype=torch.float32)  # 当前位置
    current_vel = torch.tensor(base_lin_vel, dtype=torch.float32)  # 当前线速度
    global_pos_diff = (target_pos - current_position)[:2]  # 全局位置差（x, y方向）
    global_pos_diff_direction = global_pos_diff / torch.norm(global_pos_diff)  # 全局位置差的方向
    spd = torch.abs(current_vel[0] * global_pos_diff_direction[0] + current_vel[1] * global_pos_diff_direction[1])  # 当前速度在目标方向上的投影
    total_spd += spd.item()  # 更新累计速度
    if spd.item() > max_spd:
        max_spd = spd.item()  # 更新最大速度
    reward_vel = torch.exp(-torch.abs(spd - 5.0) * 0.5)  # 速度奖励，鼓励速度接近5.0

    # 旋转奖励
    base_orn_tensor = torch.tensor(base_orn, dtype=torch.float32)  # 当前姿态四元数
    r_reach = torch.quaternion.as_rotation_matrix(torch.quaternion.quaternion(base_orn_tensor[0], base_orn_tensor[1], base_orn_tensor[2], base_orn_tensor[3]))  # 四元数转换为旋转矩阵
    yaw_reach = torch.atan2(r_reach[1, 0], r_reach[0, 0])  # 计算偏航角
    reward_rotation_current = torch.exp((torch.cos(yaw_reach) * global_pos_diff_direction[0] +
                                         torch.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 2.0)  # 旋转奖励，鼓励机器人朝向目标方向

    # 距离奖励
    pos_diff_len = torch.norm(global_pos_diff).item()  # 当前位置差的长度
    reward_dist_current = (pos_diff_len - last_pos_diff_len) * 100  # 距离奖励，鼓励靠近目标位置
    last_pos_diff_len = pos_diff_len  # 更新上一次的位置差长度

    # 综合奖励
    reward = reward_vel * 0.0 + reward_rotation_current  # 综合速度和旋转奖励
    reward -= reward_dist_current  # 减去距离奖励
    reward /= float(max_steps)  # 归一化奖励

    # 更新各部分奖励
    reward_rotation += reward_rotation_current.item() / float(max_steps)  # 更新旋转奖励
    reward_dist -= reward_dist_current / float(max_steps)  # 更新距离奖励

    return reward.item(), total_spd, max_spd, last_pos_diff_len, reward_rotation, reward_dist  # 返回计算后的奖励值和其他更新后的参数

#@torch.jit.script
def _compute_joystick_reward(
    base_pos: list, 
    base_lin_vel: list, 
    base_orn: list, 
    target_pos: torch.Tensor, 
    total_spd: float, 
    max_spd: float, 
    target_spd: float, 
    max_steps: int, 
    reward_rotation: float, 
    reward_vel: float
):
    """
    计算操纵杆奖励，包括速度奖励和旋转奖励。

    :param base_pos: 当前位置列表 [x, y, z]。
    :param base_lin_vel: 当前线速度列表 [vx, vy, vz]。
    :param base_orn: 当前姿态四元数列表 [x, y, z, w]。
    :param target_pos: 目标位置的张量。
    :param total_spd: 当前累计的速度。
    :param max_spd: 当前最大速度。
    :param target_spd: 目标速度。
    :param max_steps: 最大步数。
    :param reward_rotation: 当前旋转奖励。
    :param reward_vel: 当前速度奖励。
    :return: 计算后的奖励值，更新后的总速度，最大速度，更新后的旋转奖励，更新后的速度奖励。
    """

    # 速度奖励
    current_position = torch.tensor(base_pos, dtype=torch.float32)  # 当前位置
    current_vel = torch.tensor(base_lin_vel, dtype=torch.float32)  # 当前线速度
    global_pos_diff = (target_pos - current_position)[:2]  # 全局位置差（x, y方向）
    global_pos_diff_direction = global_pos_diff / torch.norm(global_pos_diff)  # 全局位置差的方向
    spd = torch.abs(current_vel[0] * global_pos_diff_direction[0] + current_vel[1] * global_pos_diff_direction[1])  # 当前速度在目标方向上的投影
    total_spd += spd.item()  # 更新累计速度
    if spd.item() > max_spd:
        max_spd = spd.item()  # 更新最大速度
    reward_vel_current = torch.exp(-torch.abs(spd - target_spd))  # 速度奖励，鼓励速度接近目标速度

    # 旋转奖励
    base_orn_tensor = torch.tensor(base_orn, dtype=torch.float32)  # 当前姿态四元数
    r_reach = torch.quaternion.as_rotation_matrix(torch.quaternion.quaternion(base_orn_tensor[0], base_orn_tensor[1], base_orn_tensor[2], base_orn_tensor[3]))  # 四元数转换为旋转矩阵
    yaw_reach = torch.atan2(r_reach[1, 0], r_reach[0, 0])  # 计算偏航角
    reward_rotation_current = torch.exp((torch.cos(yaw_reach) * global_pos_diff_direction[0] +
                                         torch.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 5.0)  # 旋转奖励，鼓励机器人朝向目标方向

    # 综合奖励
    reward = reward_vel_current * reward_rotation_current  # 综合速度和旋转奖励
    reward /= float(max_steps)  # 归一化奖励

    # 更新各部分奖励
    reward_rotation += reward_rotation_current.item() / float(max_steps)  # 更新旋转奖励
    reward_vel += reward_vel_current.item() / float(max_steps)  # 更新速度奖励

    return reward.item(), total_spd, max_spd, reward_rotation, reward_vel  # 返回计算后的奖励值和其他更新后的参数

#@torch.jit.script
def _compute_avg_spd_reward(states_info: dict, target_pos: torch.Tensor, total_spd: float, max_spd: float, target_spd: float, max_steps: int, init_pos_diff_len: float, last_pos_diff_len: float, counter: int, episodic_reward: dict, done_dict: dict) -> float:
    # 获取当前位置
    current_position = torch.tensor(states_info['base_pos'], dtype=torch.float32)
    # 获取当前速度
    current_vel = torch.tensor(states_info['base_lin_vel'], dtype=torch.float32)
    # 计算全局位置差
    global_pos_diff = (target_pos - current_position)[:2]
    # 计算全局位置差的方向
    global_pos_diff_direction = global_pos_diff / torch.norm(global_pos_diff)
    # 计算速度分量
    spd = torch.abs(current_vel[0] * global_pos_diff_direction[0] + current_vel[1] * global_pos_diff_direction[1])
    # 累加总速度
    total_spd += spd.item()
    # 更新最大速度
    if spd.item() > max_spd:
        max_spd = spd.item()

    # 旋转奖励
    base_orn = torch.tensor(states_info["base_orn"], dtype=torch.float32)
    r_reach = torch.quaternion.as_rotation_matrix(torch.quaternion.quaternion(base_orn[0], base_orn[1], base_orn[2], base_orn[3]))
    yaw_reach = torch.atan2(r_reach[1, 0], r_reach[0, 0])
    # 计算旋转奖励
    reward_rotation = torch.exp((torch.cos(yaw_reach) * global_pos_diff_direction[0] +
                                 torch.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 5.0)

    # 距离奖励
    pos_diff_len = torch.norm(global_pos_diff).item()
    if init_pos_diff_len is not None:
        reward_dist = (pos_diff_len - last_pos_diff_len) / init_pos_diff_len
    else:
        reward_dist = 0.0
    last_pos_diff_len = pos_diff_len

    # 缩放旋转奖励
    scaled_reward_rotation = reward_rotation.item() / float(max_steps) * 0.1
    # 缩放距离奖励
    scaled_reward_dist = - reward_dist * 0.1
    # 计算总奖励
    reward = scaled_reward_rotation * 2.0 + scaled_reward_dist
    # 累加奖励
    episodic_reward['reward_rotation'] += scaled_reward_rotation * 2.0
    episodic_reward['reward_dist'] += scaled_reward_dist

    # 判断是否完成任务
    done_cond = done_dict['done_reach']
    if done_cond:
        # 计算平均速度
        avg_spd = total_spd / counter
        # 计算平均速度奖励
        reward_avg_spd = torch.exp(-torch.abs(avg_spd - target_spd)).item()
        # 累加总奖励
        reward += reward_avg_spd
        # 累加奖励
        episodic_reward['reward_avg_spd'] += reward_avg_spd

    return reward, total_spd, max_spd, last_pos_diff_len, counter, episodic_reward