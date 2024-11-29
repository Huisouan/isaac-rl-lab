from .motion_loader import AMPLoader
import glob

def main():
    # 初始化 AMPLoader 对象
    loader = AMPLoader(
    device='cuda',
    motion_files=glob.glob(f"datasets/mocap_motions_go2/*"),
    time_between_frames=0.02,
    preload_transitions=True,
    num_preload_transitions = 1000,     
    )
    
    # 可以在这里添加更多的逻辑，例如加载数据等
    print("AMPLoader 初始化成功")

if __name__ == "__main__":
    main()