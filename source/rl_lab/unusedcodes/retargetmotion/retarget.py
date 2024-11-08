import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from go2_marker import GO2_MARKER_CFG
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort:skip
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.managers import SceneEntityCfg
from Readdata import *

from omni.isaac.lab.utils.math import subtract_frame_transforms
folder_path = "source/standalone/Mycode/retargetmotion/raw_mocap_data"

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
                    radius=0.02,
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
                    radius=0.02,
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
                    radius=0.02,
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
                    radius=0.02,
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
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.5, 0.5, 0.5)
    return VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/root_marker"))

def define_bot_markers():
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "robot_mesh": sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),

            ),
        },
    )
    return VisualizationMarkers(marker_cfg)




class MotionProcessor:
    def __init__(self, motion_data: MotionData):
        self.motion_data = motion_data
        # 检查GPU是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.default_offset = self.calculate_default_offset()
        self.apply_offset()
        self.root_pos,self.root_quat,self.root_state = self.calculate_root_state()
        self.FL_hip_toe_delta, self.FR_hip_toe_delta, self.RL_hip_toe_delta, self.RR_hip_toe_delta = self.calculate_foot_offset()

    def calculate_default_offset(self):
        """
        从所有帧中筛选出 x、y、z 距离 0 最近的点，并计算出偏移量。
        """
        # 初始化偏移量为 None
        default_offset = [0,0,0]
        min_distance = float('inf')
        frames = self.motion_data.totalframes
        if not isinstance(frames, int):
            raise ValueError("frames must be an integer")
        # 遍历所有帧
        for frame_number in range(frames):
            frame_data = self.motion_data.get_frame_without_name(frame_number)

            # 计算每帧中所有点到原点的距离
            for point in frame_data:
                z_distance = point[2]
                if z_distance < min_distance:
                    min_distance = z_distance
                    default_offset[2] = point[2]

        # 返回偏移量
        return default_offset
    
    def apply_offset(self):
        """
        将数据应用偏移量。
        """
        for name, data in self.motion_data.data.items():
            for i, xyz in enumerate(data):
                data[i] = [x - y for x, y in zip(xyz, self.default_offset)]

    
    def calculate_root_state(self):  
        """  
        根据参考关节位置计算新的根节点位置和旋转。  
        
        返回:  
        - root_pos (torch.Tensor): 计算得到的新根节点位置。  
        - root_rot (torch.Tensor): 根节点的四元数旋转表示，用于描述相对于原始坐标系的旋转。  
        """  
        # 获取数据并转换为torch.Tensor，并移到GPU上  
        pelvis_pos = torch.tensor(self.motion_data.get_data_by_name('Bip01'), dtype=torch.float64).to(self.device)  
        neck_pos = torch.tensor(self.motion_data.get_data_by_name('b__Neck'), dtype=torch.float64).to(self.device)  
        left_shoulder_pos = torch.tensor(self.motion_data.get_data_by_name('b_LeftArm'), dtype=torch.float64).to(self.device)  
        right_shoulder_pos = torch.tensor(self.motion_data.get_data_by_name('b_RightArm'), dtype=torch.float64).to(self.device)  
        left_hip_pos = torch.tensor(self.motion_data.get_data_by_name('b_LeftLegUpper'), dtype=torch.float64).to(self.device)  
        right_hip_pos = torch.tensor(self.motion_data.get_data_by_name('b_RightLegUpper'), dtype=torch.float64).to(self.device)  

        # 计算前向方向  
        forward_dir = neck_pos - pelvis_pos  
        forward_dir = F.normalize(forward_dir, p=2, dim=1)  

        # 计算左右方向  
        delta_shoulder = left_shoulder_pos - right_shoulder_pos  
        delta_hip = left_hip_pos - right_hip_pos  
        dir_shoulder = F.normalize(delta_shoulder, p=2, dim=1)  
        dir_hip = F.normalize(delta_hip, p=2, dim=1)  
        left_dir = 0.5 * (dir_shoulder + dir_hip)  

        # 确保正交性  
        up_dir = torch.cross(forward_dir, left_dir, dim=1)  
        up_dir = F.normalize(up_dir, p=2, dim=1)  
        left_dir = torch.cross(up_dir, forward_dir)  
        left_dir = F.normalize(left_dir, p=2, dim=1)  

        # 构造旋转矩阵并转换为四元数  
        n = forward_dir.shape[0]  
        root_rot_mats = []  
        for i in range(n):  
            # 构建 3x3 旋转矩阵  
            rot_mat = torch.stack([ up_dir[i],forward_dir[i], left_dir[i]], dim=1)  
            #rot_mat = rot_mat.transpose(0, 1)  
            # 转换为四元数  
            rot = R.from_matrix(rot_mat.cpu().numpy())  
            quat1 = rot.as_quat()
            y_quat = R.from_euler('y', -90, degrees=True).as_quat()
            quat2 =(R.from_quat(y_quat) * R.from_quat(quat1)).as_quat()
            x_quat = R.from_euler('x', 90, degrees=True).as_quat()
            quat =torch.tensor((R.from_quat(x_quat) * R.from_quat(quat2)).as_quat(), dtype=torch.float64, device=self.device)
                        
            # 归一化四元数  
            quat = F.normalize(quat, p=2, dim=0)  
            
            root_rot_mats.append(quat)  

        # 计算根节点位置  
        root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)  

        # 创建线速度和角速度张量，全部设置为 0  
        linear_velocity = torch.zeros((n, 3), device=self.device)  
        angular_velocity = torch.zeros((n, 6), device=self.device)  
        root_rotation = torch.stack(root_rot_mats).to(self.device)  
        rootstate = torch.cat((root_pos, root_rotation, linear_velocity, angular_velocity), dim=1)   

        return root_pos, root_rotation, rootstate 

    def calculate_foot_offset(self):
        """
        计算脚踝的偏移量。

        Args:
            robot (_type_): _description_
        """
        """
        totalframes = self.motion_data.totalframes
        if not isinstance(totalframes, int):
            raise ValueError("frames must be an integer")
        for i in range (totalframes):
            
            robot.write_root_state_to_sim(self.root_state[i])
            robot.reset()        

        """
        # 获取数据并转换为torch.Tensor，并移到GPU上
        left_shoulder = torch.tensor(self.motion_data.get_data_by_name('b_LeftArm'), dtype=torch.float64).to(self.device)
        right_shoulder = torch.tensor(self.motion_data.get_data_by_name('b_RightArm'), dtype=torch.float64).to(self.device)
        left_hip = torch.tensor(self.motion_data.get_data_by_name('b_LeftLegUpper'), dtype=torch.float64).to(self.device)
        right_hip = torch.tensor(self.motion_data.get_data_by_name('b_RightLegUpper'), dtype=torch.float64).to(self.device)

        FL_foot = torch.tensor(self.motion_data.get_data_by_name('b__LeftFinger'), dtype=torch.float64).to(self.device)
        FR_foot = torch.tensor(self.motion_data.get_data_by_name('b_RightFinger'), dtype=torch.float64).to(self.device)
        RL_foot = torch.tensor(self.motion_data.get_data_by_name('b_LeftToe002'), dtype=torch.float64).to(self.device)
        RR_foot = torch.tensor(self.motion_data.get_data_by_name('b_RightToe002'), dtype=torch.float64).to(self.device)    
        
        
        FL_hip_toe_delta = FL_foot - left_shoulder
        FR_hip_toe_delta = FR_foot - right_shoulder
        RL_hip_toe_delta = RL_foot - left_hip
        RR_hip_toe_delta = RR_foot - right_hip

        return FL_hip_toe_delta, FR_hip_toe_delta, RL_hip_toe_delta, RR_hip_toe_delta
         
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    
    
    """Runs the simulation loop."""
    
    
    
    robot = scene["robot"]
    # 定义模拟步长
    root_markers = define_root_markers()
    targetmarkers = define_markers('blue')
    sample_markers = define_markers('green')
    robotmarkers = define_markers('red')
    
    
    motion_datas = read_all_csv_motions(folder_path)
    count = 0
    """
    while simulation_app.is_running():
        sim_dt = sim.get_physics_dt()    
        if count % 1000 == 0:

            root_state = robot.data.default_root_state.clone()
            robot.write_root_state_to_sim(root_state)   
        
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)        
        jointlimit = robot.data.default_joint_limits[:,robot.find_joints('FL_calf_joint')[0],:].view(-1).to('cpu').tolist()
        
        joint_pos_des = (jointlimit[1]-jointlimit[0])*count/1000 +jointlimit[0]
        print(joint_pos_des)
        joint_pos[0][8] =joint_pos_des 
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot.find_joints('FL_calf_joint')[0])
        robot.write_joint_state_to_sim(joint_pos, joint_vel)   
        robot.write_root_state_to_sim(root_state) 
        
        scene.write_data_to_sim()   
        
        count += 1
        sim.step()
        scene.update(sim_dt)            
        if count == 1000     :
            sim_time = 0.0
            count = 0          
    """        
  
    
    
    #motion_data = MotionData("source/standalone/Mycode/retargetmotion/raw_mocap_data/dog_back_001_worldpos.csv",scale=0.005)
    for motion_data in motion_datas:
        motionprocessor = MotionProcessor(motion_data)

        FR_diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
        FR_diff_ik_controller = DifferentialIKController(FR_diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        
        FR_cfg = SceneEntityCfg("robot", joint_names=["FR.*"], body_names=['FR_hip','FR_foot'])
        FR_cfg.resolve(scene)
        FR_ee_jacobi_idx = FR_cfg.body_ids[1]
        FL_cfg = SceneEntityCfg("robot", joint_names=["FL.*"], body_names=['FL_hip','FL_foot'])
        FL_cfg.resolve(scene)
        FL_ee_jacobi_idx = FL_cfg.body_ids[1]
        RR_cfg = SceneEntityCfg("robot", joint_names=["RR.*"], body_names=['RR_hip','RR_foot'])
        RR_cfg.resolve(scene)
        RR_ee_jacobi_idx = RR_cfg.body_ids[1]
        RL_cfg = SceneEntityCfg("robot", joint_names=["RL.*"], body_names=['RL_hip','RL_foot'])
        RL_cfg.resolve(scene)
        RL_ee_jacobi_idx = RL_cfg.body_ids[1]

        # Define simulation stepping
        sim_dt = sim.get_physics_dt()    
        sim_time = 0.0
        count = 0    
        
        FR_ik_commands = torch.zeros(scene.num_envs, FR_diff_ik_controller.action_dim, device=robot.device)
        # Simulate physics
        while simulation_app.is_running():
            if not isinstance(motion_data.totalframes, int):
                raise ValueError("frames must be an integer")    
            for frame in range(motion_data.totalframes):
                
                """
                
                
                """          
                robot.write_root_pose_to_sim(motionprocessor.root_state[frame][0:7])
                
                FR_ik_commands[:] = motionprocessor.FL_hip_toe_delta[frame]+robot.data.body_pos_w[:, FR_cfg.body_ids[0], 0:3]
                if frame == 0:
                    FR_diff_ik_controller.reset()
                
                # 当前末端执行器的位置和姿态  
                ee_pose_w = robot.data.body_pos_w[:, FR_cfg.body_ids[1], 0:3]
               
                ee_quat_w = robot.data.body_quat_w[:, FR_cfg.body_ids[1], 0:4]
                
                targetpos = FR_ik_commands[:]
                print(robot.data.body_pos_w[:, FR_cfg.body_ids[0], 0:3])
                FR_diff_ik_controller.reset()
                FR_diff_ik_controller.set_command(FR_ik_commands,ee_pose_w,ee_quat_w)
                
                jacobian = robot.root_physx_view.get_jacobians()[:, FR_ee_jacobi_idx, :, FR_cfg.joint_ids]
                
                """Ordered names of bodies in articulation (through articulation view)."""
                body_names = robot.root_physx_view.shared_metatype.link_names
                print("Link names through articulation view: ", body_names)      
                
                
                # 获取机器人的根节点世界坐标姿态
                root_pose_w = robot.data.root_state_w[:, 0:7]
                # 获取当前关节位置
                joint_pos = robot.data.joint_pos[:, FR_cfg.joint_ids]


                # 计算关节指令
                joint_pos_des = FR_diff_ik_controller.compute(ee_pose_w,ee_quat_w, jacobian, joint_pos) 
                      
                # 应用动作
                robot.set_joint_position_target(joint_pos_des, joint_ids=FR_cfg.joint_ids)
                scene.write_data_to_sim()
                
                sim.step()
                scene.update(sim_dt)               

                # 获取帧数据并确保是 PyTorch 张量
                marker_indices = torch.arange(sample_markers.num_prototypes).repeat(len(motion_data.header))
                root_marker_indices = torch.arange(root_markers.num_prototypes)   
                
                translation = torch.tensor(motion_data.get_frame_without_name(count))
                sample_markers.visualize(translation, None,None, marker_indices)
                root_translation = torch.tensor(motionprocessor.root_state[frame][0:3]).unsqueeze(0)
                root_rotation = torch.tensor(motionprocessor.root_state[frame][3:7]).unsqueeze(0)            
        
                robottrans = robot.data.body_pos_w.squeeze(0)
                botorientation = robot.data.body_quat_w.squeeze(0)
                bot_marker_indices = torch.arange(robotmarkers.num_prototypes).repeat(len(robottrans))
                root_markers.visualize(root_translation,root_rotation, None, root_marker_indices)
                robotmarkers.visualize(robottrans, botorientation, None, bot_marker_indices)
                
                
                
                targetmarkers.visualize(targetpos,None,None,root_marker_indices)
                
                
                count += 1
                if count == motion_data.totalframes:
                    print(f"[INFO]: Simulated {count} frames")
                    count = 0
                    break


def main():
    
    
    
    """Main function."""
    # Load kit helper
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
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
