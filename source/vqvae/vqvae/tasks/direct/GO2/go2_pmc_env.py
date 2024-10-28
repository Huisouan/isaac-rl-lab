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
from ....datasets.go2_model import GO2_MARKER_CFG
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip
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
class PMCEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s =30.0
    decimation = 2
    num_actions = 12
    num_observations = 207
    num_states = 0
    action_scale =100
    
    kp = 50.0
    kd = 0.5
    
    
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=5.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker: ArticulationCfg = GO2_MARKER_CFG.replace(prim_path="/World/envs/env_.*/Marker")
    #contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


class PMCEnv(DirectRLEnv):
    cfg: PMCEnvCfg

    def __init__(self, cfg: PMCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = cfg.action_scale
        
        self.motiondata = MotionData("source/vqvae/data/go2")
        #init data index
        self.pmc_data_frameinplay = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.pmc_data_maxtime = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.float32)
        self.pmc_data_selected = torch.zeros(self.scene.num_envs, device=self.device,dtype=torch.int)
        #recore history observation
        self.last_observation = None
        self.second_last_observation = None
        
        
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
        self.robot.set_joint_effort_target(self.actions) 
        #self.robot.set_joint_position_target(self.actions)
        #update marker
        frame = self.motiondata.get_frame_batch(self.pmc_data_selected,self.pmc_data_frameinplay)
        rootstate = self.motiondata.root_state_w(frame)
        rootstate[:,:2] = rootstate[:,:2] + self.scene.env_origins[:,:2]
        self.marker.write_root_state_to_sim(rootstate)
        self.marker.write_joint_state_to_sim(self.motiondata.joint_position_w(frame),self.motiondata.joint_velocity_w(frame))
        
    def _get_observations(self) -> dict:      
        obs = torch.cat(#45 in total
            (
                
                self.robot.data.joint_pos,#12
                self.robot.data.joint_vel,#12
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self.actions,#12
            ),
            dim=-1,
        )
        #init last_observation and second_last_observation
        if self.last_observation is None:
            self.last_observation = obs
        if self.second_last_observation is None:
            self.second_last_observation = obs
        
        robot_state = self.robot.data.root_state_w
        robot_state[:, :2] = robot_state[:, :2] - self.scene.env_origins[:, :2]
        
        # 获取数据集get dataset
        dataset = self.motiondata.get_frame_batch_by_timelist(#72
            self.pmc_data_selected,
            self.pmc_data_frameinplay,
            robot_state
        )
        observation = torch.cat([obs,dataset], dim=1)
        # concat data and obs as observation
        observations = {"policy": observation}
        # update last_observation and second_last_observation
        self.second_last_observation = self.last_observation
        self.last_observation = obs
    
        return observations

    def _get_rewards(self) -> torch.Tensor:

        reference_frame = self.motiondata.get_frame_batch(
        self.pmc_data_selected, 
        self.pmc_data_frameinplay)  
        # base position
        root_state = self.motiondata.root_state_w(reference_frame)
        root_state[:, :2] = root_state[:, :2] + self.scene.env_origins[:, :2] 
   
        root_pos = root_state[:, :3] 

        root_orn = root_state[:, 3:7]
        # base velocities
        lin_vel = root_state[:,7:10]
        ang_vel = root_state[:,10:13]
        #joint position
        joint_pos = self.motiondata.joint_position_w(reference_frame)
        joint_vel = self.motiondata.joint_velocity_w(reference_frame)   
        
        #foot_position
        foot_positions = self.motiondata.foot_position_w(reference_frame)
        foot_positions[:,0:2] = foot_positions[:,0:2] + self.scene.env_origins[:, 0:2]
        foot_positions[:,3:5] = foot_positions[:,3:5] + self.scene.env_origins[:, 0:2]
        foot_positions[:,6:8] = foot_positions[:,6:8] + self.scene.env_origins[:, 0:2]
        foot_positions[:,9:11] = foot_positions[:,9:11] + self.scene.env_origins[:, 0:2]
        
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
            robot_joint_pos = self.robot.data.joint_pos, 
            robot_joint_vel= self.robot.data.joint_vel,
            robot_root_lin_vel_w= self.robot.data.root_lin_vel_w,
            robot_root_ang_vel_w = self.robot.data.root_ang_vel_w,
            robot_foot_positions = self.robot.data.body_pos_w[:,self.robot_foot_id[0],:].view(self.num_envs,-1),      
        )
            
        self.root_position_reward = root_position_reward
        self.tracking_position_key_body_reward = tracking_position_key_body_reward
        total_reward = joint_angle_reward + joint_velocity_reward + root_position_reward + root_velocity_reward + tracking_position_key_body_reward
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        #threshold for reward threshold
        root_threshold = 0.0000000001
        run_out_of_data = self.pmc_data_frameinplay >= self.pmc_data_maxtime
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
        
        # set into the physics simulation
        self.robot.write_root_state_to_sim(root_state,env_ids=env_ids)
        # set into the physics simulation
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids) 
        
        self.pmc_data_frameinplay[env_ids] = rand_frame.to(device=self.device).to(dtype=torch.float)
        self.pmc_data_selected[env_ids] = data_idx.to(device=self.device).to(dtype=torch.int)
        self.pmc_data_maxtime[env_ids] = data_length.to(device=self.device).to(dtype=torch.float)
        
        self.second_last_observation = self.last_observation = None

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
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
            self.pmc_data_frameinplay = self.pmc_data_frameinplay + 1

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reward_buf = self._get_rewards()
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras




@torch.jit.script
def compute_rewards(            

            lin_vel,
            ang_vel,
            joint_pos,
            joint_vel,
            foot_positions,
            robot_joint_pos,
            robot_joint_vel,
            root_pos_error,
            root_quat_error ,
            robot_root_lin_vel_w,
            robot_root_ang_vel_w,
            robot_foot_positions,
            ):
    # clamp joint pos to limits
    """
    joint_pos_limits = data.soft_joint_pos_limits
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    # clamp joint vel to limits
    joint_vel_limits = data.soft_joint_vel_limits
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)   
    """

    joint_angle_reward = 0.6 * torch.exp(-1 * torch.sum((robot_joint_pos - joint_pos) ** 2, dim=1))
    joint_vel_sum = -0.1 * torch.sum((robot_joint_vel - joint_vel) ** 2, dim=1)
    joint_velocity_reward = 0.1 * torch.exp(joint_vel_sum)

    bodyposition_l2 = -40 * torch.sum((robot_foot_positions - foot_positions) ** 2, dim=1)
    tracking_position_key_body_reward = 0.1 * torch.exp(bodyposition_l2)

    root_pos_error_sum = torch.sum(root_pos_error ** 2, dim=1)
    root_quat_error_sum = torch.sum(root_quat_error ** 2, dim=1)


    exp_part = -20 * root_pos_error_sum - 10 * root_quat_error_sum

    root_position_reward = 0.15 * torch.exp(exp_part)

    root_velocity_reward = 0.1 * torch.exp(
        -2 * torch.sum((robot_root_lin_vel_w - lin_vel) ** 2, dim=1)
        -0.1 * torch.sum((robot_root_ang_vel_w - ang_vel) ** 2, dim=1)
    )
    
    return (joint_angle_reward,
            joint_velocity_reward ,
            root_position_reward ,
            root_velocity_reward,
            tracking_position_key_body_reward)