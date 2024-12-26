from rl_lab.assets.motion_loader import AMPLoader
import glob
amp_motion_files = glob.glob(f"datasets/mocap_motions_go2/*")
amp_loader = AMPLoader(
                device='cuda',
                motion_files=amp_motion_files,
                time_between_frames=0.02,)

print(amp_loader)