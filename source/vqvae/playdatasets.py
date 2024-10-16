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
from source.vqvae.vqvae.datasets.go2_model import GO2_MARKER_CFG
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
from vqvae.datasets.motionload import MotionData
from omni.isaac.lab.utils.math import quat_rotate
from vqvae.utils import PMC_UTILS_DIR
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
        env_ids = torch.tensor([0], device='cuda')
        
        data = MotionData("source/Mycode/data")
        dt = 1/120

        for traj_idx in range(len(data.data_length)):
            frame_num = 0
            while frame_num< data.data_tensors[traj_idx].shape[0]:
                print(data.data_names[traj_idx])
                frames = data.data_tensors[traj_idx][frame_num].unsqueeze(0)
                root_state_w = data.root_state_w(frames)
                robot.write_root_state_to_sim(root_state_w)

                joint_pos = data.joint_position_w(frames)
                joint_vel = data.joint_velocity_w(frames)
    
                
                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                scene.update(dt)
                

                frame_num+=1
            

                sim.step()

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
