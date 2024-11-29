from rl_lab.assets.motionload import MotionData
import glob

def main():
    # 初始化 AMPLoader 对象
    loader = MotionData(
        data_dir = "datasets/mocap_motions_go2",
        datatype="amp",
        file_type="txt",
        env_step_duration = 0.005,    
    )
    
    # 可以在这里添加更多的逻辑，例如加载数据等
    print("AMPLoader 初始化成功")

if __name__ == "__main__":
    main()