import time
from unitree_sdk2py.core.channel import  ChannelFactoryInitialize

from unitree_sdk2py.go2.low_level.go2_pd_sim2sim import Go2_SIM2SIM
from unitree_bridge.himloco import Himloco

from unitree_bridge.config.algo.algocfg import HIMConfig
from unitree_bridge.config.robot.bot_cfg import GO2

# 默认网络接口名称
default_network = "lo"
def main():
    Algocfg = HIMConfig
    Botcfg = GO2
    model = Himloco(Algocfg,Botcfg)
    target_interval = 50
    while True:
        start_time = time.time()  # 记录循环开始时间

        imu_state, motor_state = go2.return_obs()
        velocity_commands = [0, 0, 0]
        action = model.forward(imu_state, motor_state, velocity_commands)
        end_time = time.time()  # 记录循环结束时间
        elapsed_time = end_time - start_time  # 计算实际花费的时间

        sleep_time = max(0, target_interval - elapsed_time)  # 计算需要睡眠的时间
        time.sleep(sleep_time)  # 休眠
        #赋值
        go2.extent_targetPos = action


    
    
if __name__ == "__main__":
    ChannelFactoryInitialize(1,default_network)

    # 创建Custom对象
    go2 = Go2_SIM2SIM()
    # 启动Custom对象
    go2.Start()    
    main()