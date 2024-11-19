# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import torch

from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from rl_lab.rsl_rl.utils.kinematics import urdf
from rl_lab.rsl_rl.datasets_for_txt.motion_loader import AMPLoader


class ManagerBasedRLAmpEnv(ManagerBasedRLEnv, gym.Env):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg, render_mode=render_mode)

        self.chain_ee = []
        for ee_name in self.cfg.ee_names:
            with open(self.cfg.urdf_path, "rb") as urdf_file:
                urdf_content = urdf_file.read()
                chain_ee_instance = urdf.build_serial_chain_from_urdf(urdf_content, ee_name).to(device=self.device)
                self.chain_ee.append(chain_ee_instance)

        if self.cfg.reference_state_initialization:
            print("motion_files dir: ")
            print(self.cfg.amp_motion_files)
            self.amp_loader = AMPLoader(
                device=self.device,
                motion_files=self.cfg.amp_motion_files,
                time_between_frames=self.cfg.sim.dt * self.cfg.sim.render_interval,
            )

        self.num_actions = self.action_manager.total_action_dim

        self.robot = self.scene.articulations["robot"]

    """
    Properties
    """

    def get_amp_observations(self):
        """
        获取AMP环境的观测值。

        Returns:
            torch.Tensor: 观测值的张量。
        """
        obs_manager = self.observation_manager
        # 获取AMP组中的所有观测项名称
        group_term_names = obs_manager._group_obs_term_names["AMP"]
        # 初始化一个字典来存储每个观测项的值
        group_obs = dict.fromkeys(group_term_names, None)
        # 将观测项名称和配置项打包成元组
        obs_terms = zip(group_term_names, obs_manager._group_obs_term_cfgs["AMP"])
        
        # 遍历每个观测项，计算其值并进行后处理
        for name, term_cfg in obs_terms:
            # 计算观测项的值
            obs: torch.Tensor = term_cfg.func(obs_manager._env, **term_cfg.params).clone()
            # 如果配置中有噪声，添加噪声
            if term_cfg.noise:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            # 如果配置中有裁剪范围，进行裁剪
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            # 如果配置中有缩放因子，进行缩放
            if term_cfg.scale:
                obs = obs.mul_(term_cfg.scale)
            # TODO: 引入延迟和滤波模型
            # 参考: https://robosuite.ai/docs/modules/sensors.html#observables
            # 将计算后的值存储到字典中
            group_obs[name] = obs

        # Isaac Sim 使用广度优先的关节顺序，而 Isaac Gym 使用深度优先的关节顺序
        # 因此需要重新排序关节位置和速度
        joint_pos = group_obs["joint_pos"]
        joint_vel = group_obs["joint_vel"]
        joint_pos = self.amp_loader.reorder_from_isaacsim_to_isaacgym_tool(joint_pos)
        joint_vel = self.amp_loader.reorder_from_isaacsim_to_isaacgym_tool(joint_vel)
        
        # 计算脚部位置
        foot_pos = []
        with torch.no_grad():
            for i, chain_ee in enumerate(self.chain_ee):
                # 使用前向运动学计算每个脚部的位置
                foot_pos.append(chain_ee.forward_kinematics(joint_pos[:, i * 3 : i * 3 + 3]).get_matrix()[:, :3, 3])
        foot_pos = torch.cat(foot_pos, dim=-1)
        
        # 获取基座线速度、角速度和Z轴位置
        base_lin_vel = group_obs["base_lin_vel"]
        base_ang_vel = group_obs["base_ang_vel"]
        z_pos = group_obs["base_pos_z"]
        
        # 按照指定顺序拼接所有观测值
        # joint_pos(0-11) foot_pos(12-23) base_lin_vel(24-26) base_ang_vel(27-29) joint_vel(30-41) z_pos(42)
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        terminal_amp_states = self.get_amp_observations()[reset_env_ids]
        self.extras["reset_env_ids"] = reset_env_ids
        self.extras["terminal_amp_states"] = terminal_amp_states
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
