# 导入所需的时间模块
import time
# 导入系统模块，用于处理命令行参数
import sys
import os
import threading

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
#setattr(args_cli, 'headless', True)
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import glob
import numpy as np
import torch
from rl_lab.datasets.go2_model import GO2_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.markers.config import RED_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg

from unitree_sdk2py.core.channel import  ChannelFactoryInitialize

from unitree_sdk2py.go2.low_level.go2_pd_control import Go2_PD_Control,get_key,process_key
# 默认网络接口名称
default_network = 'enp0s31f6'

model_joint_order = (
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)
go2_joint_current_order = (
    "FR_hip",   # hip
    "FR_thigh", # thigh
    "FR_calf",  # calf
    "FL_hip",   # hip
    "FL_thigh", # thigh
    "FL_calf",  # calf
    "RR_hip",   # hip
    "RR_thigh", # thigh
    "RR_calf",  # calf
    "RL_hip",   # hip
    "RL_thigh", # thigh
    "RL_calf"   # calf
)

def joint_reorder(tensor, src_order, tgt_order):
    assert len(src_order) == len(tgt_order), f"src_order and tgt_order must have same length."
    index_map = [src_order.index(joint) for joint in tgt_order]
    tensor_new = tensor[index_map]
    return tensor_new

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot:ArticulationCfg = GO2_MARKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def define_markers(color):
    if color == "red":
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/MyMarkers",
            markers={               
                "sphere": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),                  
            }   
        )
        return VisualizationMarkers(marker_cfg)
    elif color == "green":
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/MyMarkers",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),                  
            }
        )
        return VisualizationMarkers(marker_cfg)
    elif color == "blue":
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/MyMarkers",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),                  
            }
        )
        return VisualizationMarkers(marker_cfg)
    else:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/MyMarkers",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                ),                  
            }
        )
        return VisualizationMarkers(marker_cfg)
   
   
def define_root_markers ():
    
    """
    坐标箭头，红色是x，朝前，绿色是y，朝左，蓝色是z，朝上。
    Returns:
        _type_: _description_
    """
    frame_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
    return VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/root_marker"))


def go2_obs_process(imu_state,motor_state):

    quaternion = torch.zeros(4)
    for i in range(4):
        quaternion[i] = imu_state.quaternion[i]

    joint_pos = torch.zeros(12)
    joint_vel = torch.zeros(12)
    for i in range(12):
        joint_pos[i] = motor_state[i].q
        joint_vel[i] = motor_state[i].dq
    
    return quaternion,joint_pos,joint_vel


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,go2:Go2_PD_Control):
    while simulation_app.is_running():
        """Runs the simulation loop."""
        sim_dt = sim.get_physics_dt()
        robot = scene["robot"]
        while True:
            imu_state,motor_state = go2.return_obs()      
            quaternion,joint_pos,joint_vel = go2_obs_process(imu_state,motor_state)
            joint_pos = joint_reorder(joint_pos,go2_joint_current_order,model_joint_order)
            joint_vel = joint_reorder(joint_vel,go2_joint_current_order,model_joint_order)

            root_state = robot.data.default_root_state.clone()
            root_state[0,3:7] = quaternion
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            sim.step()
            scene.update(sim_dt)
            
            


def control_loop(go2):
    while True:
        timestart = time.time()
        key = get_key()
        if key is not None:
            go2.control_mode, reset_mode = process_key(key)
            if reset_mode:
                go2.reset()
            if key == 'q':
                time.sleep(1)
                print("Done!")
                sys.exit(-1)
    


if __name__ == "__main__":
    """Main function."""
    # Load kit helper
    # run the main function
    # 创建Custom对象
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # 等待用户确认
    input("Press Enter to continue...")

    # 初始化通道工厂
    ChannelFactoryInitialize(0,default_network)    
    
    go2 = Go2_PD_Control()
    # 初始化Custom对象
    go2.Init()
    # 启动Custom对象
    go2.Start()    
    
    # 创建线程
    control_thread = threading.Thread(target=control_loop, args=(go2,))
    # 启动线程
    control_thread.start()       
    
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene,go2)
    # close sim app
    simulation_app.close()
