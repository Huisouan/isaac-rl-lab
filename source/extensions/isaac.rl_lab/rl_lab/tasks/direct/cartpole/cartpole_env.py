# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from rl_lab.assets.motionload import MotionData

@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    #重置最大时长
    episode_length_s = 30
    action_scale = 100.0  # [N]
    num_actions = 1
    num_observations = 100
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #marker_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Marker")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]



class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel
        self.motiondata = MotionData("source/rl_lab/data/cartpole")
        
        #初始化数据集采样的时间
        
        self.pmc_data_frameinplay = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.pmc_data_maxtime = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.pmc_data_selected = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.int)  
        
        
        self.last_observation = None
        self.second_last_observation = None
        self.observation = None
              
    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        #self.marker = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
    def _get_observations(self) -> dict:
        self.pmc_data_frameinplay +=1
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        
        if self.last_observation is None:
            self.last_observation = obs
        if self.second_last_observation is None:
            self.second_last_observation = obs
        dataset = self.motiondata.get_frame_batch_by_timelist_cartpole(#64
            self.pmc_data_selected,
            self.pmc_data_frameinplay,
            obs
        )
        obsservation = torch.cat((self.second_last_observation, self.last_observation, obs, dataset), dim=1)
        observations = {"policy": obsservation}
        
        self.second_last_observation = self.last_observation
        self.last_observation = obs
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        frame = self.motiondata.get_frame_batch(self.pmc_data_selected,self.pmc_data_frameinplay)
        total_reward = compute_rewards(
            frame,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        data_out = self.pmc_data_frameinplay>self.pmc_data_maxtime
        
        
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        
        

        return out_of_bounds, time_out|data_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)
        frames,data_idx,rand_frame,data_length = self.motiondata.get_random_frame_batch(len(env_ids))
        

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]        
        
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        joint_pos = frames[:, [2,0]]
        joint_vel =frames[:, [3,1]]
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel,None, env_ids = env_ids)  
    

        self.pmc_data_frameinplay[env_ids] = rand_frame.to(device=self.device).to(dtype=torch.float)
        self.pmc_data_selected[env_ids] = data_idx.to(device=self.device).to(dtype=torch.int)
        self.pmc_data_maxtime[env_ids] = data_length.to(device=self.device).to(dtype=torch.float)
        
        #更新observations
        joint_state = self.cartpole.data.root_state_w 
        joint_state[:,:2]-= self.scene.env_origins[:,:2]        
        
        self.second_last_observation = self.last_observation = None        


def compute_rewards(
            frame,
            joint_pos_pole,
            joint_vel_pole,
            joint_pos_cart,
            joint_vel_cart,
            reset_terminated,
        ):
    joint_pos_pole_l2 = torch.exp(-1 * (frame[:,0] - joint_pos_pole) ** 2)
    joint_vel_pole_l2 = torch.exp(-1 * (frame[:,1] - joint_vel_pole) ** 2)
    joint_pos_cart_l2 = torch.exp(-1 * (frame[:,2] - joint_pos_cart) ** 2)
    joint_vel_cart_l2 = torch.exp(-1 * (frame[:,3] - joint_vel_cart) ** 2)
    reward = 10 * joint_pos_pole_l2 + joint_vel_pole_l2 + joint_pos_cart_l2 + joint_vel_cart_l2
    
    return reward