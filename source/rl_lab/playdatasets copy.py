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
from .rl_lab.datasets.go2_model import GO2_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
folder_path = "source/standalone/Mycode/data"
from rl_lab.datasets.motion_loader import DATALoader
from omni.isaac.lab.utils.math import quat_rotate
from rl_lab.utils import PMC_UTILS_DIR
import csv
import os
import csv
class CSVWriter:
    def __init__(self, abs_path):
        self.abs_path = abs_path
        self.filename = self._get_csv_filename()

    def _get_csv_filename(self):
        """
        从绝对路径中提取文件名，并将 .txt 后缀改为 .csv。
        """
        base_name = os.path.basename(self.abs_path)
        return os.path.splitext(base_name)[0] + '.csv'

    def initialize_csv(self):
        """
        初始化 CSV 文件并写入列标题。
        """
        self.headers = [
            *[f"rootstate{i}" for i in range(13)],
            *[f"joint_pos_{i}" for i in range(12)],
            *[f"joint_vel_{i}" for i in range(12)],
            *[f"foot_pos_{i}_{axis}" for i in range(4) for axis in ["x", "y", "z"]],
            *[f"foot_vel_{i}_{axis}" for i in range(4) for axis in ["x", "y", "z","rx","ry","rz"]]
        ]
        with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if self.headers:
                writer.writerow(self.headers)

    def write_data(self, data_row):
        """
        将数据行写入 CSV 文件。
        
        :param data_row: 列表或元组，包含一行数据
        """
        with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_row)


def output_motion(frames, out_filename, motion_weight, frame_duration):
    with open(out_filename, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write('"FrameDuration": ' + str(frame_duration) + ",\n")
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": ' + str(motion_weight) + ",\n")
        f.write("\n")

        f.write('"Frames":\n')

        f.write("[")
        for i in range(frames.shape[0]):
            curr_frame = frames[i]

            if i != 0:
                f.write(",")
            f.write("\n  [")

            for j in range(frames.shape[1]):
                curr_val = curr_frame[j]
                if j != 0:
                    f.write(", ")
                f.write("%.5f" % curr_val)

            f.write("]")

        f.write("\n]")
        f.write("\n}")

    return


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
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.5, 0.5, 0.5)
    return VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/root_marker"))


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    while simulation_app.is_running():
        """Runs the simulation loop."""
        robot = scene["robot"]
        motion_files=glob.glob(f"{PMC_UTILS_DIR}/motion_files/mocap_motions_go2/*")
        env_ids = torch.tensor([0], device='cuda')
        
        data = DATALoader("source/Mycode/data")
        dt = 1/120
        times = 0.0
        for traj_idx in range(data.num_motions):
            while times< data.trajectory_lens[traj_idx]:
                frames = data.get_full_frame_at_time(traj_idx, times)
                positions = data.get_root_pos(frames)
                orientations = data.get_root_rot(frames)
                lin_vel = quat_rotate(orientations, data.get_linear_vel(frames))
                ang_vel = quat_rotate(orientations, data.get_angular_vel(frames))
                velocities = torch.cat([lin_vel, ang_vel], dim=-1)
                robot.write_root_pose_to_sim(
                    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
                )
                robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
                joint_pos = data.get_joint_pose(frames)
                joint_vel = data.get_joint_vel(frames)

                joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids]
                joint_pos = torch.clamp(joint_pos, min=joint_pos_limits[:, :, 0], max=joint_pos_limits[:, :, 1])
    
                
                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                scene.update(dt)
                foot_id = robot.find_bodies(['FL_foot','FR_foot','RL_foot','RR_foot'])
                
                sim_rootstate = robot.data.root_state_w[0]
                sim_joint_pos = robot.data.joint_pos[0]
                sim_joint_vel = robot.data.joint_vel[0]
                foot_positions = robot.data.body_pos_w[:,foot_id[0],:][0].view(-1)      
                foot_velocities = robot.data.body_vel_w[:,foot_id[0],:][0].view(-1)   
                datarow = torch.cat([sim_rootstate, sim_joint_pos, sim_joint_vel, foot_positions,foot_velocities])
                datarow_list = datarow.tolist()
                writer.write_data(datarow_list)
                

                times += dt
            

                sim.step()
            times = 0.0

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
